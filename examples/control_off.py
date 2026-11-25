"""Run with `python examples/control_off.py` to try negative controls."""

from guppylang import guppy
from guppylang.std.builtins import array, control, control_off, dagger, nat, output
from guppylang.std.quantum import discard_array, h, measure, qubit, s, x


@guppy.unitary
class custom_x:
    @guppy(unitary=True)
    def __call__(target: qubit) -> None:
        x(target)

    @guppy
    def ctrl_daggered[n: nat](target: qubit, controls: array[qubit, n]) -> None:
        # This method uses ordinary positive controls. X is its own inverse.
        with control(controls):
            x(target)


@guppy
def main() -> None:
    # Both controls and targets start in |0>.
    zero_control = qubit()
    zero_target = qubit()
    one_control = qubit()
    one_target = qubit()
    x(one_control)  # Prepare this control in |1>.

    with control_off(zero_control):
        x(zero_target)  # Runs because the control is |0>.

    with control_off(one_control):
        x(one_target)  # Does not run because the control is |1>.

    output("zero_control", measure(zero_control).read())
    output("zero_target", measure(zero_target).read())
    output("one_control", measure(one_control).read())
    output("one_target", measure(one_target).read())

    # Multiple arguments: every control must be |0> for the block to run.
    c1, c2, c3 = qubit(), qubit(), qubit()
    all_zero_target = qubit()
    with control_off(c1, c2, c3):
        x(all_zero_target)

    x(c2)
    one_set_target = qubit()
    with control_off(c1, c2, c3):
        x(one_set_target)  # Does not run: c2 is |1>.

    output("multiple_all_zero", measure(all_zero_target).read())
    output("multiple_one_set", measure(one_set_target).read())
    output("multiple_c1", measure(c1).read())
    output("multiple_c2", measure(c2).read())
    output("multiple_c3", measure(c3).read())

    # An array works the same way: all of its elements are controls.
    controls = array(qubit(), qubit(), qubit())
    array_zero_target = qubit()
    with control_off(controls):
        x(array_zero_target)

    x(controls[1])
    array_one_target = qubit()
    with control_off(controls):
        x(array_one_target)  # Does not run: controls[1] is |1>.

    output("array_all_zero", measure(array_zero_target).read())
    output("array_one_set", measure(array_one_target).read())
    discard_array(controls)

    # Combining modifiers applies the inverse of the block when the control is |0>.
    dagger_zero_control = qubit()
    dagger_one_control = qubit()
    dagger_zero_target = qubit()
    dagger_one_target = qubit()
    x(dagger_one_control)
    h(dagger_zero_target)
    s(dagger_zero_target)
    x(dagger_one_target)

    with control_off(dagger_zero_control), dagger:
        s(dagger_zero_target)  # S-dagger cancels the earlier S.
    with control_off(dagger_one_control), dagger:
        x(dagger_one_target)  # Skipped because the control is |1>.

    # Convert the phase difference into a measurable bit.
    h(dagger_zero_target)
    output("dagger_zero_control_target", measure(dagger_zero_target).read())
    output("dagger_one_control_target", measure(dagger_one_target).read())
    output("dagger_zero_control", measure(dagger_zero_control).read())
    output("dagger_one_control", measure(dagger_one_control).read())

    # The custom controlled-dagger implementation also works with control_off.
    custom_zero_control = qubit()
    custom_one_control = qubit()
    custom_zero_target = qubit()
    custom_one_target = qubit()
    x(custom_one_control)
    with control_off(custom_zero_control), dagger:
        custom_x(custom_zero_target)
    with control_off(custom_one_control), dagger:
        custom_x(custom_one_target)

    output("custom_zero_target", measure(custom_zero_target).read())
    output("custom_one_target", measure(custom_one_target).read())
    output("custom_zero_control", measure(custom_zero_control).read())
    output("custom_one_control", measure(custom_one_control).read())


if __name__ == "__main__":
    result = main.emulator(n_qubits=5).statevector_sim().with_shots(1).run()
    print(result.collated_shots())  # noqa: T201
