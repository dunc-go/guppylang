import pytest

from guppylang import guppy
from guppylang.std.builtins import array, control, dagger, mixed_controls, nat
from guppylang.std.quantum import discard, discard_array, h, measure, qubit, s, x, z


@pytest.mark.parametrize(
    ("c0", "c1", "m0", "m1"),
    [
        (c0, c1, m0, m1)
        for c0 in (False, True)
        for c1 in (False, True)
        for m0 in (False, True)
        for m1 in (False, True)
    ],
)
def test_mixed_controls_truth_table(run_int_fn, c0, c1, m0, m1):
    @guppy
    def main(c0: bool, c1: bool, m0: bool, m1: bool) -> int:
        controls = array(qubit(), qubit())
        target = qubit()
        if c0:
            x(controls[0])
        if c1:
            x(controls[1])
        mask = array(m0, m1)
        with mixed_controls(controls, mask):
            x(target)
        result = measure(target).read()
        q0 = controls.take(0)
        q1 = controls.take(1)
        bit0 = measure(q0).read()
        bit1 = measure(q1).read()
        discard_array(controls)
        return (1 if result else 0) + (2 if bit0 else 0) + (4 if bit1 else 0)

    expected = int(c0 == m0 and c1 == m1) + 2 * int(c0) + 4 * int(c1)
    run_int_fn(main, expected=expected, args=[c0, c1, m0, m1], num_qubits=3)


def test_mixed_controls_superposition(run_int_fn):
    @guppy
    def main() -> int:
        controls = array(qubit())
        target = qubit()
        h(controls[0])
        x(target)
        with mixed_controls(controls, array(False)):
            z(target)
        h(controls[0])
        result = measure(controls.take(0)).read()
        discard_array(controls)
        discard(target)
        return 1 if result else 0

    run_int_fn(main, expected=1, num_qubits=2)


def test_mixed_controls_empty_array(run_int_fn):
    @guppy
    def main() -> int:
        controls: array[qubit, 0] = array()
        mask: array[bool, 0] = array()
        target = qubit()
        with mixed_controls(controls, mask):
            x(target)
        result = measure(target).read()
        discard_array(controls)
        return 1 if result else 0

    run_int_fn(main, expected=1, num_qubits=1)


def test_mixed_controls_generic_nested(run_int_fn):
    @guppy(controllable=True)
    def apply[n: nat](
        controls: array[qubit, n], mask: array[bool, n], q: qubit
    ) -> None:
        with mixed_controls(controls, mask):
            x(q)

    @guppy
    def main() -> int:
        outer = qubit()
        controls = array(qubit(), qubit())
        target = qubit()
        x(outer)
        x(controls[1])
        with control(outer):
            apply(controls, array(False, True), target)
        result = measure(target).read()
        discard(outer)
        discard_array(controls)
        return 1 if result else 0

    run_int_fn(main, expected=1, num_qubits=4)


def test_mixed_controls_dagger(run_int_fn):
    @guppy
    def main() -> int:
        controls = array(qubit())
        target = qubit()
        h(target)
        s(target)
        with mixed_controls(controls, array(False)), dagger:
            s(target)
        h(target)
        result = measure(target).read()
        discard_array(controls)
        return 1 if result else 0

    run_int_fn(main, expected=0, num_qubits=2)


def test_mixed_controls_under_dagger(run_int_fn):
    @guppy(daggerable=True)
    def apply(controls: array[qubit, 1], mask: array[bool, 1], target: qubit) -> None:
        with mixed_controls(controls, mask):
            s(target)

    @guppy
    def main() -> int:
        controls = array(qubit())
        target = qubit()
        h(target)
        s(target)
        with dagger:
            apply(controls, array(False), target)
        h(target)
        result = measure(target).read()
        discard_array(controls)
        return 1 if result else 0

    run_int_fn(main, expected=0, num_qubits=2)


def test_mixed_controls_runtime_measurement_mask(run_int_fn):
    @guppy
    def main() -> int:
        source = qubit()
        h(source)
        bit = measure(source).read()
        controls = array(qubit())
        target = qubit()
        mask = array(bit)
        with mixed_controls(controls, mask):
            x(target)
        result = measure(target).read()
        mask_bit = mask[0]
        discard_array(controls)
        return 1 if result != mask_bit else 0

    run_int_fn(main, expected=1, num_qubits=3)


def test_mixed_controls_mask_used_inside_block(run_int_fn):
    @guppy
    def main() -> int:
        controls = array(qubit())
        mask = array(False)
        target = qubit()
        with mixed_controls(controls, mask):
            if not mask[0]:
                x(target)
        result = measure(target).read()
        discard_array(controls)
        return 1 if result else 0

    run_int_fn(main, expected=1, num_qubits=2)


def test_mixed_controls_subscripted_mask(run_int_fn):
    @guppy
    def main() -> int:
        controls = array(qubit(), qubit())
        masks = array(array(False, True), array(True, False))
        target = qubit()
        x(controls[1])
        with mixed_controls(controls, masks[0]):
            x(target)
        result = measure(target).read()
        first_mask_bit = masks[0][0]
        discard_array(controls)
        return 1 if result and not first_mask_bit else 0

    run_int_fn(main, expected=1, num_qubits=3)


def test_mixed_controls_custom_ctrl_daggered(run_int_fn):
    @guppy.unitary
    class gate:
        @guppy(unitary=True)
        def __call__(target: qubit) -> None:
            pass

        @guppy
        def ctrl_daggered[n: nat](target: qubit, controls: array[qubit, n]) -> None:
            with control(controls):
                x(target)

    @guppy
    def main() -> int:
        controls = array(qubit(), qubit())
        target = qubit()
        x(controls[1])
        with mixed_controls(controls, array(False, True)), dagger:
            gate(target)
        result = measure(target).read()
        discard_array(controls)
        return 1 if result else 0

    run_int_fn(main, expected=1, num_qubits=3)


@pytest.mark.parametrize("mixed_first", [False, True])
def test_mixed_controls_with_different_sized_control_group(run_int_fn, mixed_first):
    @guppy
    def main(mixed_first: bool) -> int:
        positive = qubit()
        controls = array(qubit(), qubit())
        target = qubit()
        x(positive)
        x(controls[1])
        mask = array(False, True)
        if mixed_first:
            with mixed_controls(controls, mask), control(positive):
                x(target)
        else:
            with control(positive), mixed_controls(controls, mask):
                x(target)
        result = measure(target).read()
        positive_bit = measure(positive).read()
        first_bit = measure(controls.take(0)).read()
        second_bit = measure(controls.take(1)).read()
        discard_array(controls)
        return (
            (1 if result else 0)
            + (2 if positive_bit else 0)
            + (4 if first_bit else 0)
            + (8 if second_bit else 0)
        )

    run_int_fn(main, expected=11, args=[mixed_first], num_qubits=4)
