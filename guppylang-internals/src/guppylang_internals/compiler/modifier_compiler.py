"""Hugr generation for modifiers."""

from hugr import Wire
from hugr import tys as ht
from hugr.ops import CallIndirect, ExtOp

from guppylang_internals.ast_util import get_type
from guppylang_internals.checker.core import SubscriptAccess, contains_subscript
from guppylang_internals.checker.modifier_checker import non_copyable_front_others_back
from guppylang_internals.compiler.builder import pure
from guppylang_internals.compiler.cfg_compiler import compile_cfg
from guppylang_internals.compiler.core import CompilerContext, DFContainer
from guppylang_internals.compiler.expr_compiler import ExprCompiler
from guppylang_internals.metadata.common import add_unitary_metadata
from guppylang_internals.nodes import CheckedModifiedBlock, PlaceNode
from guppylang_internals.std._internal.compiler.array import (
    array_clone,
    array_map,
    array_new,
    array_to_std_array,
    standard_array_type,
    std_array_to_array,
    unpack_array,
)
from guppylang_internals.std._internal.compiler.tket_exts import (
    MODIFIER_EXTENSION,
    QUANTUM_EXTENSION,
)
from guppylang_internals.tys import Effect
from guppylang_internals.tys.builtin import int_type, is_array_type
from guppylang_internals.tys.ty import InputFlags, UnitaryFlags


def _flip_qubit(dfg: DFContainer, qubit: Wire) -> Wire:
    """Apply X to a control qubit."""
    op = ExtOp(QUANTUM_EXTENSION.get_op("X"), ht.FunctionType([ht.Qubit], [ht.Qubit]))
    return dfg.builder.add_op(pure(op), qubit)


def _flip_control_array(dfg: DFContainer, array: Wire, length: ht.TypeArg) -> Wire:
    """Apply X to each element of a Guppy control array."""
    if isinstance(length, ht.BoundedNatArg):
        qubits = unpack_array(dfg.builder, array)
        flipped = [_flip_qubit(dfg, qubit) for qubit in qubits]
        return dfg.builder.add_op(array_new(ht.Qubit, length.n), *flipped)

    # A generic-length array cannot be unpacked at compile time. Map a unitary
    # function over it; only the array access can panic, since X is pure.
    flip = dfg.builder.define_function("__control_off_x", [ht.Qubit], [ht.Qubit])
    add_unitary_metadata(flip._raw.hugr[flip].metadata, UnitaryFlags.Unitary.value)
    op = ExtOp(QUANTUM_EXTENSION.get_op("X"), ht.FunctionType([ht.Qubit], [ht.Qubit]))
    result = flip.add_op(pure(op), flip.inputs()[0])
    flip.set_outputs(result)
    flip_fn = dfg.builder.load_function(flip, ht.FunctionType([ht.Qubit], [ht.Qubit]))
    map_op, _ = array_map(ht.Qubit, length, ht.Qubit)
    return dfg.builder.add_op((map_op, [Effect.PANIC]), array, flip_fn)


def _flip_mixed_controls(dfg: DFContainer, array: Wire, mask: list[Wire]) -> Wire:
    """Flip each control whose corresponding mask bit is false."""
    qubits = unpack_array(dfg.builder, array)
    flipped: list[Wire] = []
    for qubit, bit in zip(qubits, mask, strict=True):
        conditional = dfg.builder.add_conditional(bit, qubit)
        false_case = conditional.add_case(0)
        op = ExtOp(
            QUANTUM_EXTENSION.get_op("X"), ht.FunctionType([ht.Qubit], [ht.Qubit])
        )
        false_case.set_outputs(false_case.add_op(pure(op), false_case.inputs()[0]))
        true_case = conditional.add_case(1)
        true_case.set_outputs(true_case.inputs()[0])
        [result] = conditional.outputs()
        flipped.append(result)
    return dfg.builder.add_op(array_new(ht.Qubit, len(flipped)), *flipped)


def compile_modified_block(
    modified_block: CheckedModifiedBlock,
    dfg: DFContainer,
    ctx: CompilerContext,
    expr_compiler: ExprCompiler,
) -> Wire:
    DAGGER_OP_NAME = "DaggerModifier"
    CONTROL_OP_NAME = "ControlModifier"
    POWER_OP_NAME = "PowerModifier"

    dagger_op_def = MODIFIER_EXTENSION.get_op(DAGGER_OP_NAME)
    control_op_def = MODIFIER_EXTENSION.get_op(CONTROL_OP_NAME)
    power_op_def = MODIFIER_EXTENSION.get_op(POWER_OP_NAME)

    body_ty = modified_block.ty
    # TODO: Shouldn't this be `to_hugr_poly` since it can contain
    # a variable with a generic type?
    hugr_ty = body_ty.to_hugr(ctx)
    in_out_ht = [
        fn_inp.ty.to_hugr(ctx)
        for fn_inp in body_ty.inputs
        if InputFlags.Inout in fn_inp.flags and InputFlags.Comptime not in fn_inp.flags
    ]
    other_in_ht = [
        fn_inp.ty.to_hugr(ctx)
        for fn_inp in body_ty.inputs
        if InputFlags.Inout not in fn_inp.flags
        and InputFlags.Comptime not in fn_inp.flags
    ]
    in_out_arg = ht.ListArg([t.type_arg() for t in in_out_ht])
    other_in_arg = ht.ListArg([t.type_arg() for t in other_in_ht])

    func_builder = dfg.builder.define_function(
        str(modified_block), hugr_ty.input, hugr_ty.output
    )
    add_unitary_metadata(
        func_builder._raw.hugr[func_builder].metadata,
        modified_block.ty.unitary_flags.value,
    )

    # compile body
    cfg = compile_cfg(modified_block.cfg, func_builder, func_builder.inputs(), ctx)
    func_builder.set_outputs(*cfg)
    effects = func_builder.effects

    # Add the LoadFunc node
    call = dfg.builder.load_function(func_builder, hugr_ty)

    # Captured values to be the function inputs
    captured = [v for v, _ in modified_block.captured.values()]
    captured = non_copyable_front_others_back(captured)

    # Apply modifiers
    if modified_block.has_dagger():
        dagger_ty = ht.FunctionType([hugr_ty], [hugr_ty])
        call = dfg.builder.add_op(
            pure(  # This is generation of the daggered version, not calling it (below)
                ExtOp(
                    dagger_op_def,
                    dagger_ty,
                    [in_out_arg, other_in_arg],
                )
            ),
            call,
        )
    if modified_block.has_power():
        power_ty = ht.FunctionType([hugr_ty, int_type().to_hugr(ctx)], [hugr_ty])
        for power in modified_block.power:
            num = expr_compiler.compile(power.iter, dfg)
            call = dfg.builder.add_op(
                # This is generation of the powered version, not calling it (below)
                pure(
                    ExtOp(
                        power_op_def,
                        power_ty,
                        [in_out_arg, other_in_arg],
                    )
                ),
                call,
                num,
            )
    qubit_num_args = []
    if modified_block.has_control():
        for control in modified_block.control:
            assert control.qubit_num is not None
            qubit_num: ht.TypeArg
            if isinstance(control.qubit_num, int):
                qubit_num = ht.BoundedNatArg(control.qubit_num)
            else:
                qubit_num = control.qubit_num.to_arg().to_hugr(ctx)
            qubit_num_args.append(qubit_num)
            std_array = standard_array_type(ht.Qubit, qubit_num)

            # Control adds a standard qubit array before the body inputs.
            input_fn_ty = hugr_ty
            output_fn_ty = ht.FunctionType(
                [std_array, *hugr_ty.input], [std_array, *hugr_ty.output]
            )
            # Compilation of the controlled version is pure, not calling it (below)
            op = pure(
                ExtOp(
                    control_op_def,
                    ht.FunctionType([input_fn_ty], [output_fn_ty]),
                    [qubit_num, in_out_arg, other_in_arg],
                )
            )
            call = dfg.builder.add_op(op, call)
            # Update types: later modifiers see the newly wrapped function type
            in_out_arg = ht.ListArg([std_array.type_arg(), *in_out_arg.elems])
            hugr_ty = output_fn_ty

    # Compile control expressions and pack individual controls into arrays.
    ctrl_args: list[Wire] = []
    masks: list[list[Wire] | None] = []
    for i, control in enumerate(modified_block.control):
        if is_array_type(get_type(control.ctrl[0])):
            control_array = expr_compiler.compile(control.ctrl[0], dfg)
            mask = None
            if control.mask is not None:
                mask_array = expr_compiler.compile(control.mask, dfg)
                if isinstance(control.mask, PlaceNode):
                    original, mask_array = dfg.builder.add_op(
                        array_clone(ht.Bool, qubit_num_args[i]), mask_array
                    )
                    dfg[control.mask.place] = original
                    if subscript := contains_subscript(control.mask.place):
                        assert subscript.setitem_call is not None
                        dfg[subscript.setitem_call.value_var] = dfg[subscript]
                        expr_compiler.visit(subscript.setitem_call.call)
                mask = unpack_array(dfg.builder, mask_array)
                control_array = _flip_mixed_controls(dfg, control_array, mask)
            elif control.on_zero:
                control_array = _flip_control_array(
                    dfg, control_array, qubit_num_args[i]
                )
            masks.append(mask)
            # Note we assume this can panic (as any array_to_std_array) if the array has
            # any missing/borrowed elements. If we could guarantee that the controls do
            # not come from/though a custom-modifier (that might `array.take()` qubits),
            # then this array_to_std_array will not panic.
            control_array = dfg.builder.add_op(
                array_to_std_array(ht.Qubit, qubit_num_args[i]), control_array
            )
            ctrl_args.append(control_array)
        else:
            masks.append(None)
            cs = [expr_compiler.compile(c, dfg) for c in control.ctrl]
            if control.on_zero:
                cs = [_flip_qubit(dfg, c) for c in cs]
            control_array = dfg.builder.add_op(
                array_new(ht.Qubit, len(control.ctrl)), *cs
            )
            # TODO this array_to_std_array never panics; we could mark it so explicitly,
            # or better, create a new std_array in a single op.
            control_array = dfg.builder.add_op(
                array_to_std_array(ht.Qubit, qubit_num_args[i]), *control_array
            )
            ctrl_args.append(control_array)

    # Wires corresponding to captured values before the indirect call.
    args = [dfg[v] for v in captured]

    # Call the modified block.
    call = dfg.builder.add_op(
        (CallIndirect(), effects),
        call,
        *reversed(ctrl_args),
        *args,
    )
    outports = iter(call)

    # Unpack returned controls
    control_subscripts: list[SubscriptAccess] = []
    # Each control modifier prepends its controls to the function signature,
    # so the last modifier's controls are returned first.
    for i in reversed(range(len(modified_block.control))):
        control = modified_block.control[i]
        outport = next(outports)
        if is_array_type(get_type(control.ctrl[0])):
            control_array = dfg.builder.add_op(
                std_array_to_array(ht.Qubit, qubit_num_args[i]), outport
            )
            if (mask := masks[i]) is not None:
                control_array = _flip_mixed_controls(dfg, control_array, mask)
            elif control.on_zero:
                control_array = _flip_control_array(
                    dfg, control_array, qubit_num_args[i]
                )
            c = control.ctrl[0]
            assert isinstance(c, PlaceNode)
            dfg[c.place] = control_array
            if subscript := contains_subscript(c.place):
                control_subscripts.append(subscript)
        else:
            control_array = dfg.builder.add_op(
                std_array_to_array(ht.Qubit, qubit_num_args[i]), outport
            )
            # TODO this unpack never panics; we could mark it so explicitly,
            # or better, unpack the std_array directly.
            unpacked = unpack_array(dfg.builder, control_array)
            if control.on_zero:
                unpacked = [_flip_qubit(dfg, c) for c in unpacked]
            for c, new_c in zip(control.ctrl, unpacked, strict=False):
                assert isinstance(c, PlaceNode)
                dfg[c.place] = new_c
                if subscript := contains_subscript(c.place):
                    control_subscripts.append(subscript)

    for arg in captured:
        # captured inout values are returned after the control arrays.
        if not arg.ty.copyable:
            dfg[arg] = next(outports)

    for subscript in control_subscripts:
        # Restore a returned control value using the setter from type checking.
        assert subscript.setitem_call is not None
        dfg[subscript.setitem_call.value_var] = dfg[subscript]
        expr_compiler.visit(subscript.setitem_call.call)

    return call
