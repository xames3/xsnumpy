"""\
xsNumPy Utilities
=================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 25 2024
Last updated on: Thursday, December 05 2024

This module provides utility functions designed to streamline and
enhance the development process within the xsNumPy library. These
utilities include decorators and helper functions that improve code
organization, modularity, and introspection. By encapsulating reusable
patterns and functionality, this module promotes clean, maintainable,
and efficient code throughout the library.
"""

from __future__ import annotations

import typing as t

__all__: list[str] = [
    "as_strided",
]


def as_strided(
    shape: t.Sequence[int],
    itemsize: int,
) -> tuple[int, ...]:
    """Calculate memory strides for a multi-dimensional array.

    Strides represent the step size in bytes to traverse along each
    dimension of an array in memory. This function computes the strides
    for a given array shape and element size, assuming a row-major
    (C-style) memory layout. In this layout, the last axis is contiguous
    in memory and changes the fastest.

    .. note::

        [1] This function mimics the stride computation logic of NumPy's
            internal memory model for arrays. In particular, it aligns
            with the default C-style (row-major) layout used in NumPy
            when creating arrays.
        [2] Unlike `numpy.lib.stride_tricks.as_strided`, this
            implementation focuses solely on computing strides based on
            shape and element size without introducing functionality to
            create new views of arrays with custom strides.

    NumPy's version allows for advanced memory manipulation, while this
    implementation is a simpler, type-safe calculation utility.

    :param shape: The dimensions of the array, given as a sequence of
        integers. Each integer represents the size of the array along
        that dimension.
    :param itemsize: The size, in bytes, of a single array element. This
        is typically determined by the data type (e.g., 4 bytes for a
        32-bit integer).
    :return: A tuple of integers representing the strides, in bytes, for
        each dimension of the array. The order of the strides
        corresponds to the input shape.
    """
    strides: list[int] = []
    stride: int = itemsize
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))
