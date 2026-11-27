from guppylang import guppy
from guppylang.std.builtins import array, mixed_controls
from guppylang.std.quantum import qubit


@guppy
def invalid(control: qubit, mask: array[bool, 1]) -> None:
    with mixed_controls(control, mask):
        pass


invalid.compile()
