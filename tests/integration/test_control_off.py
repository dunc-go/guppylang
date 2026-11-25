import pytest
from hugr import tys as ht
from hugr.build.function import Module

from guppylang_internals.compiler.builder import FunctionBuilder
from guppylang_internals.compiler.core import DFContainer
from guppylang_internals.compiler.modifier_compiler import _flip_control_array
from guppylang_internals.std._internal.compiler.array import array_type

from guppylang.decorator import guppy
from guppylang.std.array import array
from guppylang.std.builtins import control, control_off, dagger
from guppylang.std.num import nat
from guppylang.std.quantum import cx, discard, discard_array, h, measure, qubit, x, z


@pytest.mark.parametrize(("initial", "expected"), [(False, 1), (True, 2)])
def test_control_off_preserves_control(run_int_fn, initial, expected):
    @guppy
    def main(initial: bool) -> int:
        c = qubit()
        target = qubit()
        if initial:
            x(c)
        with control_off(c):
            x(target)
        target_bit = measure(target).read()
        control_bit = measure(c).read()
        return (1 if target_bit else 0) + (2 if control_bit else 0)

    run_int_fn(main, expected=expected, num_qubits=2, args=[initial])


def test_control_off_superposition(run_int_fn):
    @guppy
    def main() -> int:
        c = qubit()
        target = qubit()
        x(target)
        h(c)
        with control_off(c):
            z(target)
        h(c)
        result = measure(c).read()
        discard(target)
        return 1 if result else 0

    run_int_fn(main, expected=1, num_qubits=2)


@pytest.mark.parametrize(("second_on", "expected"), [(False, 1), (True, 0)])
def test_control_off_multiple(run_int_fn, second_on, expected):
    @guppy
    def main(second_on: bool) -> int:
        c1 = qubit()
        c2 = qubit()
        target = qubit()
        if second_on:
            x(c2)
        with control_off(c1, c2):
            x(target)
        result = measure(target).read()
        discard(c1)
        discard(c2)
        return 1 if result else 0

    run_int_fn(main, expected=expected, num_qubits=3, args=[second_on])


def test_control_off_generic_array(run_int_fn):
    @guppy
    def apply[n: nat](controls: array[qubit, n], target: qubit) -> None:
        with control_off(controls):
            x(target)

    @guppy
    def main() -> int:
        controls = array(qubit(), qubit())
        target = qubit()
        apply(controls, target)
        result = measure(target).read()
        discard_array(controls)
        return 1 if result else 0

    run_int_fn(main, expected=1, num_qubits=3)


def test_control_off_symbolic_array_lowering(validate):
    # Guppy normally monomorphizes array lengths before lowering. Exercise the
    # symbolic-length path directly so that its generated HUGR is also validated.
    param = ht.BoundedNatParam()
    length = ht.VariableArg(0, param)
    ty = array_type(ht.Qubit, length)
    module = Module()
    raw = module.define_function("test", [ty], [ty], type_params=[param])
    builder = FunctionBuilder(raw)
    dfg = DFContainer(builder, None)  # type: ignore[arg-type]
    flipped = _flip_control_array(dfg, builder.inputs()[0], length)
    builder.set_outputs(flipped)
    validate(module.hugr, export=False)


@pytest.mark.parametrize(("second_on", "expected"), [(False, 1), (True, 0)])
def test_control_off_array(run_int_fn, second_on, expected):
    @guppy
    def main(second_on: bool) -> int:
        controls = array(qubit(), qubit())
        target = qubit()
        if second_on:
            x(controls[1])
        with control_off(controls):
            x(target)
        result = measure(target).read()
        discard_array(controls)
        return 1 if result else 0

    run_int_fn(main, expected=expected, num_qubits=3, args=[second_on])


def test_control_off_subscript(run_int_fn):
    @guppy
    def main() -> int:
        qs = array(qubit())
        target = qubit()
        with control_off(qs[0]):
            x(target)
        result = measure(target).read()
        discard_array(qs)
        return 1 if result else 0

    run_int_fn(main, expected=1, num_qubits=2)


def test_control_off_mixed_controls(run_int_fn):
    @guppy
    def main() -> int:
        positive = qubit()
        negative = qubit()
        target = qubit()
        x(positive)
        with control(positive), control_off(negative):
            x(target)
        result = measure(target).read()
        discard(positive)
        discard(negative)
        return 1 if result else 0

    run_int_fn(main, expected=1, num_qubits=3)


def test_control_off_custom_controlled(run_int_fn):
    @guppy.unitary
    class gate:
        @guppy
        def __call__(target: qubit) -> None:
            x(target)

        @guppy
        def controlled[n: nat](target: qubit, controls: array[qubit, n]) -> None:
            cx(controls[0], target)

    @guppy
    def main() -> int:
        c = qubit()
        target = qubit()
        with control_off(c):
            gate(target)
        result = measure(target).read()
        discard(c)
        return 1 if result else 0

    run_int_fn(main, expected=1, num_qubits=2)


def test_control_off_nested_modifiers(run_int_fn):
    @guppy
    def main() -> int:
        outer = qubit()
        inner = qubit()
        target = qubit()
        x(outer)
        with control(outer), dagger:
            with control_off(inner):
                x(target)
        result = measure(target).read()
        discard(outer)
        discard(inner)
        return 1 if result else 0

    run_int_fn(main, expected=1, num_qubits=3)


@pytest.mark.parametrize(("outer_on", "expected"), [(False, 0), (True, 1)])
def test_control_off_generic_array_under_control(run_int_fn, outer_on, expected):
    @guppy(controllable=True)
    def apply[n: nat](controls: array[qubit, n], target: qubit) -> None:
        with control_off(controls):
            x(target)

    @guppy
    def main(outer_on: bool) -> int:
        outer = qubit()
        controls = array(qubit(), qubit())
        target = qubit()
        if outer_on:
            x(outer)
        with control(outer):
            apply(controls, target)
        result = measure(target).read()
        discard(outer)
        discard_array(controls)
        return 1 if result else 0

    run_int_fn(main, expected=expected, num_qubits=4, args=[outer_on])
