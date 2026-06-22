"""Reference CPU kernels used by early tests."""

from __future__ import annotations


def add_vectors(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    return tuple(a + b for a, b in zip(left, right))

