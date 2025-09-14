"""Additional utilities to increase test count and exercise typing."""
from __future__ import annotations

from typing import List


def cumulative_sums(nums: List[int]) -> List[int]:
    """Return running cumulative sums of a list of integers."""
    out: List[int] = []
    total = 0
    for n in nums:
        total += n
        out.append(total)
    return out


def is_sorted(nums: List[int]) -> bool:
    """Return True if list is non-decreasing."""
    return all(x <= y for x, y in zip(nums, nums[1:]))
