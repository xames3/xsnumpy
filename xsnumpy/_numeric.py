"""\
xsNumPy Array Functions
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, December 06 2024
Last updated on: Saturday, December 07 2024

This module provides core functionality to create and manipulate xsNumPy
arrays.
"""

from __future__ import annotations

import math
import typing as t

from xsnumpy import array_function_dispatch
from xsnumpy import ndarray
from xsnumpy._typing import DTypeLike
from xsnumpy._typing import _OrderKACF
from xsnumpy._typing import _ShapeLike


@array_function_dispatch
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
    if not isinstance(shape, t.Iterable):
        shape = (shape,)
    return ndarray(shape, dtype, order=order)


@array_function_dispatch
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


@array_function_dispatch
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


@array_function_dispatch
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


@array_function_dispatch
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


@array_function_dispatch
def eye(
    N: int,
    M: int | None = None,
    dtype: DTypeLike | None = None,
    order: None | _OrderKACF = None,
) -> ndarray:
    """Create a 2-D identity matrix with ones on the main diagonal and
    zeros elsewhere.

    The `eye` function generates a square array of the specified size
    with ones on its main diagonal (from the top-left to the bottom-
    right corner) and zeros elsewhere.

    :param N: The number of rows of the identity matrix.
    :param M: The number of columns of the identity matrix, defaults to
        `None` meaning M == N.
    :param dtype: The desired data type of the output array, defaults to
        `None`.
    :param order: The desired memory layout for the output array,
        defaults to `None`.
    :return: A square identity matrix of shape `(N, N)`.
    """
    if N <= 0:
        raise ValueError("Size must be a positive integer")
    if M is None:
        M = N
    arr = zeros((N, M), dtype, order)
    for idx in range(N):
        arr[idx, idx] = 1
    return arr


identity = array_function_dispatch(eye)


@array_function_dispatch
def tri(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: DTypeLike | None = None,
    order: None | _OrderKACF = None,
) -> ndarray:
    """Generate a lower triangular matrix with ones below and on the
    main diagonal, and zeros elsewhere.

    The function creates a two-dimensional array with the specified
    dimensions. The elements below and on the main diagonal are set to
    1, while the elements above the diagonal are set to 0.

    :param N: The number of rows of the identity matrix.
    :param M: The number of columns of the identity matrix, defaults to
        `None` meaning M == N.
    :param k: Diagonal offset, defaults to 0.
    :param dtype: The desired data type of the output array, defaults to
        `None`.
    :param order: The desired memory layout for the output array,
        defaults to `None`.
    :return: A 2D array of shape `(N, M)` where the elements below and
        on the main diagonal are 1, and the elements above the diagonal
        are 0.
    """
    if N <= 0:
        raise ValueError("Size must be a positive integer")
    if M is None:
        M = N
    arr = zeros((N, M), dtype, order)
    for idx in range(N):
        start = max(0, idx + k)
        end = min(M, idx + 1 + k)
        if start < end:
            arr[idx, start:end] = 1
    return arr


@array_function_dispatch
def diag(v: ndarray, k: int = 0) -> ndarray:
    """Extract the diagonal of a 2D array or construct a diagonal array.

    If the input array `v` is 2-dimensional, this function extracts the
    elements along the diagonal offset by `k`. If `v` is 1-dimensional,
    it constructs a 2D array with the elements of `v` as the diagonal
    offset by `k`.

    :param v: Input array. Must be 1D or 2D.
    :param k: Diagonal offset, defaults to 0.
    :return: If `v` is 2D, returns a 1D array containing the elements of
        the specified diagonal. If `v` is 1D, returns a 2D array with
        the specified diagonal populated.
    :raise ValueError: If the input array `v` is not 1D or 2D.
    """
    if v.ndim == 1:
        size = v.shape[0]
        result = zeros((size + abs(k), size + abs(k)), dtype=v.dtype)
        for idx in range(size):
            result[(idx + k, idx) if k < 0 else (idx, idx + k)] = v[idx]
        return result
    elif v.ndim == 2:
        rows, cols = v.shape
        if k >= 0:
            start, end = k, min(rows, cols + k)
            return ndarray(
                (end - start,),
                dtype=v.dtype,
                buffer=v,
                offset=k * v.strides[1],
            )
        else:
            start, end = -k, min(rows + k, cols)
            return ndarray(
                (end - start,),
                dtype=v.dtype,
                buffer=v,
                offset=-k * v.strides[0],
            )
    else:
        raise ValueError("Input array must be 1D or 2D")


@array_function_dispatch
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
        raise TypeError("arange() requires stop to be specified")
    elif c_args > 3:
        raise TypeError("Too many input arguments")
    start, stop, step = (
        (0.0, args[0], 1.0)
        if c_args == 1
        else (args[0], args[1], 1.0) if c_args == 2 else args
    )
    if step == 0:
        raise ValueError("Step size must not be zero")
    size = max(0, math.ceil((stop - start) / step))
    result = empty((size,), dtype=dtype)
    result[:] = [start + idx * step for idx in range(size)]
    return result
