from guppylang import guppy
from guppylang.std.builtins import array, mixed_controls
from guppylang.std.quantum import qubit


@guppy
def invalid(controls: array[qubit, 2], mask: array[int, 2]) -> None:
    with mixed_controls(controls, mask):
        pass


invalid.compile()
