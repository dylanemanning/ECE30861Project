from __future__ import annotations

import pytest

from myproject.core import add, factorial, maybe_divide


@pytest.mark.parametrize("a,b,expected", [(1, 1, 2), (2, 3, 5), (0, 0, 0), (-1, 1, 0)])
def test_add_param(a: int, b: int, expected: int) -> None:
    assert add(a, b) == expected


@pytest.mark.parametrize("n,expected", [(0, 1), (1, 1), (3, 6), (6, 720)])
def test_factorial_param(n: int, expected: int) -> None:
    assert factorial(n) == expected


@pytest.mark.parametrize("a,b,expected", [(4, 2, 2.0), (5, 2, 2.5), (0, 1, 0.0)])
def test_maybe_divide_param(a: int, b: int, expected: float) -> None:
    assert maybe_divide(a, b) == expected
