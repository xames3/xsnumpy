"""\
xsNumPy Utilities
=================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 25 2024
Last updated on: Thursday, December 05 2024

This module provides utility functions designed to streamline and
enhance the development process within the xsNumPy library. These
utilities include helper functions that improve code organization,
modularity, and introspection. These utilities focus on enhancing code
modularity and enabling type-safe operations.
"""

from __future__ import annotations

import typing as t

__all__: list[str] = [
    "calc_strides",
]


def calc_strides(
    shape: t.Sequence[int],
    itemsize: int,
) -> tuple[int, ...]:
    """Calculate strides for traversing a multi-dimensional array.

    Strides determine the step size (in bytes) required to move to the
    next element along each dimension of an array in memory. This
    function calculates strides assuming a row-major (C-style) memory
    layout, where the last dimension is contiguous in memory and changes
    the fastest.

    :param shape: The dimensions of the array. Each integer specifies the
        size along a particular dimension.
    :param itemsize: The size (in bytes) of a single array element.
        Typically determined by the data type.
    :return: A tuple of integers representing the strides, in bytes, for
        each dimension of the array. The order of the strides
        corresponds to the input `shape`, starting with the first
        dimension.

    .. note::

        Similarities with NumPy::
            [1] Aligns with NumPy's C-style (row-major) memory layout
                for stride computation.
            [2] Matches the behavior of NumPy when initializing arrays
                with default settings.

        Differences from NumPy::
            [1] Unlike NumPy's `as_strided`, this function does not
                create array views or manipulate memory. It focuses
                solely on stride computation.
    """
    strides: list[int] = []
    stride: int = itemsize
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))
