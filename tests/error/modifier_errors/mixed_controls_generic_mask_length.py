from guppylang import guppy
from guppylang.std.builtins import array, mixed_controls, nat
from guppylang.std.quantum import qubit


@guppy
def apply[n: nat](controls: array[qubit, n], mask: array[bool, n]) -> None:
    with mixed_controls(controls, mask):
        pass


@guppy
def invalid(controls: array[qubit, 2], mask: array[bool, 3]) -> None:
    apply(controls, mask)


invalid.compile()
