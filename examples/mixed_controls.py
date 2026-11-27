"""Demonstrate `mixed_controls(controls, mask)`.

The controls and mask are arrays of the same length. Each `True` mask entry
controls on |1> (control on); each `False` entry controls on |0> (control off).
The block runs when every control matches its corresponding mask entry.

Run with `python examples/mixed_controls.py`.
"""

from guppylang import guppy
from guppylang.std.builtins import array, control, dagger, mixed_controls, nat, output
from guppylang.std.quantum import discard_array, h, measure, qubit, s, x


@guppy.unitary
class custom_s:
    @guppy(unitary=True)
    def __call__(target: qubit) -> None:
        s(target)

    @guppy
    def ctrl_daggered[n: nat](target: qubit, controls: array[qubit, n]) -> None:
        # Custom implementations receive ordinary positive controls.
        with control(controls), dagger:
            s(target)


@guppy
def main() -> None:
    # Four controls match the mask |0111>, so the target flips.
    controls = array(qubit(), qubit(), qubit(), qubit())
    mask = array(False, True, True, True)
    target = qubit()
    x(controls[1])
    x(controls[2])
    x(controls[3])
    with mixed_controls(controls, mask):
        x(target)

    output("fixed_mask_target", measure(target).read())
    discard_array(controls)

    # This mask bit is only known after a runtime measurement.
    coin = qubit()
    h(coin)
    mask_bit = measure(coin).read()
    runtime_controls = array(qubit())  # Prepared in |0>.
    runtime_mask = array(mask_bit)
    runtime_target = qubit()
    with mixed_controls(runtime_controls, runtime_mask):
        x(runtime_target)

    # The target flips if mask_bit is False and stays |0> if it is True.
    output("runtime_mask_bit", mask_bit)
    output("runtime_mask_target", measure(runtime_target).read())
    discard_array(runtime_controls)

    # A matching mixed mask also selects a custom controlled-dagger method.
    custom_controls = array(qubit(), qubit())
    custom_mask = array(False, True)
    custom_target = qubit()
    x(custom_controls[1])
    h(custom_target)
    s(custom_target)
    with mixed_controls(custom_controls, custom_mask), dagger:
        custom_s(custom_target)  # S-dagger cancels the earlier S.
    h(custom_target)
    output("custom_dagger_target", measure(custom_target).read())
    discard_array(custom_controls)


if __name__ == "__main__":
    results = (
        main.emulator(n_qubits=5).statevector_sim().with_seed(42).with_shots(8).run()
    )
    for shot in results.collated_shots():
        print(shot)  # noqa: T201
