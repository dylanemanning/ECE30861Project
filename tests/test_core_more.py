from __future__ import annotations

from typing import List

from myproject.core_utils import cumulative_sums, is_sorted


def test_cumulative_empty() -> None:
    assert cumulative_sums([]) == []


def test_cumulative_single() -> None:
    assert cumulative_sums([5]) == [5]


def test_cumulative_multi() -> None:
    assert cumulative_sums([1, 2, 3]) == [1, 3, 6]


def test_is_sorted_true() -> None:
    assert is_sorted([1, 1, 2, 3])


def test_is_sorted_false() -> None:
    assert not is_sorted([3, 2, 1])


def test_combined_behavior() -> None:
    l: List[int] = [0, 1, 1, 2]
    assert is_sorted(l)
    assert cumulative_sums(l) == [0, 1, 2, 4]
