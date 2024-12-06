"""\
xsNumPy Core Functions
======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, December 06 2024
Last updated on: Friday, December 06 2024

This module provides core functionality to create and manipulate xsNumPy
arrays.
"""

from __future__ import annotations

import typing as t

from xsnumpy._typing import DTypeLike
from xsnumpy._typing import _OrderKACF
from xsnumpy._typing import _ShapeLike

if t.TYPE_CHECKING:
    from xsnumpy import ndarray

__all__: list[str] = [
    "arange",
    "empty",
    "eye",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
]


def empty(
    shape: _ShapeLike,
    dtype: DTypeLike | None = None,
    order: None | _OrderKACF = None,
) -> ndarray:
    """Create a new ndarray without initializing its values.

    The `empty` function returns a new `ndarray` with the specified
    shape, data type, and memory layout order. The contents of the array
    are uninitialized and may contain random data. This function is
    useful for performance-critical applications where immediate
    initialization is not required.

    :param shape: The desired shape of the array. Can be an int for
        1D arrays or a sequence of ints for multidimensional arrays.
    :param dtype: The desired data type of the array, defaults to
        `None` if not specified.
    :param order: The memory layout of the array, defaults to `None`.
    :return: An uninitialized array with the specified properties.

    .. note::

        [1] The contents of the returned array are random and should
            not be used without proper initialization.
    """
    from xsnumpy import ndarray

    return ndarray(shape, dtype, order=order)


def zeros(
    shape: _ShapeLike,
    dtype: DTypeLike | None = None,
    order: None | _OrderKACF = None,
) -> ndarray:
    """Create a new ndarray filled with zeros.

    The `zeros` function creates an ndarray with the specified shape,
    data type, and memory layout, initializing all its elements to zero.
    This function is particularly useful for scenarios requiring a blank
    array with known dimensions and type, where all elements must
    initially be zero.

    :param shape: The desired shape of the array. Can be an int for
        1D arrays or a sequence of ints for multidimensional arrays.
    :param dtype: The desired data type of the array, defaults to
        `None` if not specified.
    :param order: The memory layout of the array, defaults to `None`.
    :return: An array initialized with zeros with the specified
        properties.
    """
    return empty(shape, dtype, order)


def zeros_like(
    a: ndarray,
    dtype: DTypeLike | None = None,
    order: None | _OrderKACF = None,
) -> ndarray:
    """Create a new array with the same shape and type as a given array,
    filled with zeros.

    The `zeros_like` function creates an array with the same shape as
    the input array `a`. By default, the new array will have the same
    data type as `a`, but this can be overridden with the `dtype`
    parameter.

    :param a: The reference array whose shape and optionally type are
        used to create the new array.
    :param dtype: The desired data type of the new array. If `None`,
        the data type of `a` is used, defaults to `None`.
    :param order: The desired memory layout for the new array, defaults
        to `None`.
    :return: A new array filled with zeros, matching the shape of `a` and
        the specified or inherited data type and memory layout.
    """
    dtype = a.dtype if dtype is None else dtype
    return zeros(a.shape, dtype, order)


def ones(
    shape: _ShapeLike,
    dtype: DTypeLike | None = None,
    order: None | _OrderKACF = None,
) -> ndarray:
    """Create a new ndarray filled with ones.

    The `ones` function creates an ndarray with the specified shape,
    data type, and memory layout, initializing all its elements to zero.
    This function is particularly useful for scenarios requiring a blank
    array with known dimensions and type, where all elements must
    initially be zero.

    :param shape: The desired shape of the array. Can be an int for
        1D arrays or a sequence of ints for multidimensional arrays.
    :param dtype: The desired data type of the array, defaults to
        `None` if not specified.
    :param order: The memory layout of the array, defaults to `None`.
    :return: An array initialized with ones with the specified
        properties.
    """
    arr = empty(shape, dtype, order)
    arr.fill(1)
    return arr


def ones_like(
    a: ndarray,
    dtype: DTypeLike | None = None,
    order: None | _OrderKACF = None,
) -> ndarray:
    """Create a new array with the same shape and type as a given array,
    filled with ones.

    The `ones_like` function creates an array with the same shape as
    the input array `a`. By default, the new array will have the same
    data type as `a`, but this can be overridden with the `dtype`
    parameter.

    :param a: The reference array whose shape and optionally type are
        used to create the new array.
    :param dtype: The desired data type of the new array. If `None`,
        the data type of `a` is used, defaults to `None`.
    :param order: The desired memory layout for the new array, defaults
        to `None`.
    :return: A new array filled with ones, matching the shape of `a` and
        the specified or inherited data type and memory layout.
    """
    dtype = a.dtype if dtype is None else dtype
    return ones(a.shape, dtype, order)


def eye(
    size: int,
    dtype: DTypeLike | None = None,
    order: None | _OrderKACF = None,
) -> ndarray:
    """Create a 2-D identity matrix with ones on the main diagonal and
    zeros elsewhere.

    The `eye` function generates a square array of the specified size
    with ones on its main diagonal (from the top-left to the bottom-
    right corner) and zeros elsewhere. The data type and memory layout
    of the array can be customized.

    :param size: The number of rows and columns of the identity matrix.
        This determines the shape `(size, size)` of the returned array.
    :param dtype: The desired data type of the output array, defaults to
        `None`.
    :param order: The desired memory layout for the output array,
        defaults to `None`.
    :return: A square identity matrix of shape `(size, size)`.
    """
    if size <= 0:
        raise ValueError("Size must be a positive integer")
    arr = zeros((size, size), dtype, order)
    for idx in range(size):
        arr[idx, idx] = 1
    return arr


# TODO(xames3): Try to implement or support floating point step size.
def arange(*args: t.Any, dtype: DTypeLike | None = None) -> ndarray:
    """Return evenly spaced values within a given range.

    The `arange` function generates a 1-D array containing evenly spaced
    values over a specified interval. The interval is defined by the
    `start`, `stop`, and `step` arguments. It mimics the behavior of
    Python's built-in `range` function but returns an `ndarray` instead.

    :param dtype: The desired data type of the output array, defaults to
        `None`, in which case the data type is inferred.
    :return: A 1-D array of evenly spaced values.

    .. note::

        [1] Unlike NumPy's `arange`, this implementation uses integer
            only inputs and does not support floating-point step sizes.
    """
    c_args = len(args)
    if c_args == 0:
        raise TypeError("Required argument 'start' not found")
    elif c_args > 3:
        raise TypeError("Too many input arguments")
    start, stop, step = (
        (0, args[0], 1)
        if c_args == 1
        else (args[0], args[1], 1) if c_args == 2 else args
    )
    if step == 0:
        raise ValueError("Step size must not be zero")
    size = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
    result = empty((size,), dtype=dtype)
    # HACK(xames3): Below is just a hot patch to never consider floating
    # numbers. This needs to be worked on.
    result[:] = [*range(start, stop, step)]
    return result
