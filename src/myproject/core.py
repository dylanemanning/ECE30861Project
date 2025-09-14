"""Small typed utilities with simple logic to test."""
from __future__ import annotations

from typing import Optional


def add(a: int, b: int) -> int:
    return a + b


def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def maybe_divide(a: int, b: int) -> Optional[float]:
    if b == 0:
        return None
    return a / b
