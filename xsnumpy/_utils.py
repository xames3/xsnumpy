"""\
xsNumPy Utilities
=================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 25 2024
Last updated on: Friday, December 06 2024

This module provides utility functions designed to streamline and
enhance the development process within the xsNumPy library. These
utilities include helper functions that improve code organization,
modularity, and introspection. These utilities focus on enhancing code
modularity and enabling type-safe operations.
"""

from __future__ import annotations

import math
import typing as t

if t.TYPE_CHECKING:
    from xsnumpy import ndarray

__all__: list[str] = [
    "calc_size",
    "calc_strides",
    "get_step_size",
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


def calc_size(shape: t.Sequence[int]) -> int:
    """Calculate the total number of elements in an array given its
    shape.

    This function computes the product of the dimensions in the shape
    sequence, which corresponds to the total size (or number of
    elements) of a multidimensional array.

    :param shape: The dimensions of the array. Each integer specifies the
        size along a particular dimension.
    :return: The total number of elements in the array.
    """
    return math.prod(shape)


def get_step_size(view: ndarray) -> int:
    """Compute the step size to traverse an array.

    This function calculates the step size based on the strides of the
    given view. If the view is C-contiguous (i.e., the last axis of the
    array is stored contiguously in memory), the step size will be 1. IF
    the array's strides are not contiguous, it returns 0.

    :param view: The array view whose step size and contiguity are to be
        determined.
    :return: A step size indicating whether the array is C-contiguous or
        not. 1 meaning contiguous else 0.

    .. note::

        This function assumes row-major (C-style) memory layout when
        checking for contiguity.
    """
    contiguous_strides = calc_strides(view.shape, view.itemsize)
    step_size = view.strides[-1] // contiguous_strides[-1]
    strides = tuple(stride * step_size for stride in contiguous_strides)
    return step_size if view.strides == strides else 0
