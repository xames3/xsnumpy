"""\
xsNumPy Array Functions
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, December 06 2024
Last updated on: Wednesday, January 22 2025

This module provides essential array creation and initialization
utilities for the `xsnumpy` package. It contains a suite of functions
designed to construct and populate `ndarray` objects with various
patterns and values, mimicking the functionality of NumPy's core array
creation routines.

This module serves as the foundation for generating arrays with specific
shapes, patterns, and values. These utilities are essential for
initializing numerical data structures, enabling users to quickly
prototype and perform computations without the need for manual data
entry. Inspired by NumPy's array creation APIs, this module brings
similar functionality to `xsnumpy` with a focus on educational clarity,
pure Python implementation, and modular design.

The following functions are implemented in this module::

    - Array Creation Functions
    - Pattern-Based Array Functions
    - Array Transformation Functions

This module is designed to balance functionality and clarity, making it
both a practical tool and an educational resource. Key principles
guiding its implementation include::

    - Consistency: Functions follow predictable naming conventions and
      parameter usage, ensuring a seamless experience for users familiar
      with NumPy.
    - Flexibility: Support for multiple data types, shapes, and memory
      layouts.
    - Simplicity: Implementations prioritize readability and modularity,
      enabling users to explore and extend functionality with ease.
    - Educational Value: As part of the `xsnumpy` project, this module
      emphasizes the learning of array mechanics and API design.

The array creation functions in this module are ideal for::

    - Initializing arrays for numerical computations.
    - Creating test datasets for algorithm development.
    - Prototyping applications that require structured numerical data.
    - Exploring the mechanics of multidimensional array creation in Python.

The implementations in this module are not optimized for performance and
are intended for learning and exploratory purposes. For production-grade
numerical computation, consider using NumPy directly.
"""

from __future__ import annotations

import itertools
import math
import typing as t

import xsnumpy as xp
from xsnumpy import array_function_dispatch
from xsnumpy._core import _BaseDType
from xsnumpy._core import ndarray
from xsnumpy._typing import DTypeLike
from xsnumpy._typing import _ArrayType
from xsnumpy._typing import _OrderKACF
from xsnumpy._typing import _ShapeLike
from xsnumpy._utils import calc_shape_from_obj
from xsnumpy._utils import has_uniform_shape
from xsnumpy._utils import safe_range


@array_function_dispatch
def array(
    object: _ArrayType | int | float,
    dtype: DTypeLike = None,
    *,
    order: _OrderKACF = None,
) -> ndarray:
    """Create an ndarray from a Python iterable.

    This function builds an ndarray from a sequence of elements, which
    can include nested sequences to represent multidimensional data.
    The function validates input uniformity and infers the appropriate
    shape and data type.

    :param object: Input data, such as a list or tuple, to be converted
        into an ndarray. The input can have nested iterables for
        multidimensional arrays.
    :param dtype: The desired data type of the array, defaults to
        `None`, in which case the type is inferred from the input data.
    :param order: Row-major or column-major order, defaults to `None`.
    :return: A new array populated with data from the input iterable.
    :raises ValueError: If the input is not uniform in its nested
        dimensions.
    """
    if not has_uniform_shape(object):
        raise ValueError("Input data is not uniformly nested")
    shape = shape if (shape := calc_shape_from_obj(object)) else (1,)
    array_like: list[t.Any] = []

    def _flatten(data: _ArrayType | int | float) -> None:
        """Recursively flatten the input iterable."""
        if isinstance(data, t.Iterable):
            for item in data:
                _flatten(item)
        else:
            array_like.append(data)

    _flatten(object)
    if dtype is None:
        dtype = (
            xp.int64
            if all(isinstance(idx, int) for idx in array_like)
            else xp.float64
        )
    out = ndarray(shape, dtype, order=order)
    out[:] = array_like
    return out


@array_function_dispatch
def empty(
    *shape: int,
    dtype: DTypeLike = None,
    order: _OrderKACF = None,
) -> ndarray:
    """Create a new ndarray without initializing its values.

    The `empty` function returns a new `ndarray` with the specified
    shape, data type, and memory layout order. The contents of the array
    are uninitialized and contain random data, in theory but
    practically, it fills them with zeros because of `ctypes`. This
    function is useful for performance-critical applications where
    immediate initialization is not required.

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
    if len(shape) == 1:
        out = ndarray(shape[0], dtype, order=order)
        out[:] = [0.0 for _ in safe_range(shape)]
        return out
    elif len(shape) > 1:
        out = ndarray(shape, dtype, order=order)
        N = range(max(shape))
        for dim in itertools.product(N, N):
            try:
                out[dim] = 0.0
            except IndexError:
                continue
        return out
    else:
        raise TypeError(
            f"Expected a sequence of integers or a single integer, "
            f"got {shape!r}"
        )


@array_function_dispatch
def zeros(
    *shape: int,
    dtype: DTypeLike = None,
    order: _OrderKACF = None,
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
    return empty(*shape, dtype=dtype, order=order)


@array_function_dispatch
def zeros_like(
    a: ndarray,
    dtype: DTypeLike = None,
    order: _OrderKACF = None,
    shape: None | _ShapeLike = None,
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
    :param shape: The desired shape of the array. Can be an int for
        1D arrays or a sequence of ints for multidimensional arrays,
        defaults to `None`.
    :return: A new array filled with zeros, matching the shape of `a` and
        the specified or inherited data type and memory layout.
    """
    dtype = a.dtype if dtype is None else dtype
    if shape is None:
        shape = a.shape
    return zeros(*shape, dtype=dtype, order=order)


@array_function_dispatch
def ones(
    *shape: int,
    dtype: DTypeLike = None,
    order: _OrderKACF = None,
) -> ndarray:
    """Create a new ndarray filled with ones.

    The `ones` function creates an ndarray with the specified shape,
    data type, and memory layout, initializing all its elements to one.
    This function is particularly useful for scenarios requiring a blank
    array with known dimensions and type, where all elements must
    initially be ones.

    :param shape: The desired shape of the array. Can be an int for
        1D arrays or a sequence of ints for multidimensional arrays.
    :param dtype: The desired data type of the array, defaults to
        `None` if not specified.
    :param order: The memory layout of the array, defaults to `None`.
    :return: An array initialized with ones with the specified
        properties.
    """
    out = empty(*shape, dtype=dtype, order=order)
    out.fill(1)
    return out


@array_function_dispatch
def ones_like(
    a: ndarray,
    dtype: DTypeLike = None,
    order: _OrderKACF = None,
    shape: None | _ShapeLike = None,
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
    :param shape: The desired shape of the array. Can be an int for
        1D arrays or a sequence of ints for multidimensional arrays,
        defaults to `None`.
    :return: A new array filled with ones, matching the shape of `a` and
        the specified or inherited data type and memory layout.
    """
    dtype = a.dtype if dtype is None else dtype
    if shape is None:
        shape = a.shape
    return ones(*shape, dtype=dtype, order=order)


@array_function_dispatch
def full(
    *shape: int,
    fill_value: int | float,
    dtype: DTypeLike = None,
    order: _OrderKACF = None,
) -> ndarray:
    """Create a new ndarray filled with `fill_value`.

    The `full` function creates an ndarray with the specified shape,
    data type, and memory layout, initializing all its elements to
    `fill_value`. This function is particularly useful for scenarios
    requiring a blank array with known dimensions and type, where all
    elements must initially be `fill_value`.

    :param shape: The desired shape of the array. Can be an int for
        1D arrays or a sequence of ints for multidimensional arrays.
    :param fill_value: Value to be filled with in the ndarray.
    :param dtype: The desired data type of the array, defaults to
        `None` if not specified.
    :param order: The memory layout of the array, defaults to `None`.
    :return: An array initialized with ones with the specified
        properties.
    """
    out = empty(*shape, dtype=dtype, order=order)
    out.fill(fill_value)
    return out


@array_function_dispatch
def eye(
    N: int,
    M: None | int = None,
    k: int = 0,
    dtype: DTypeLike = None,
    order: _OrderKACF = None,
) -> ndarray:
    """Create a 2-D identity matrix with ones on the main diagonal and
    zeros elsewhere.

    The `eye` function generates a square array of the specified size
    with ones on its main diagonal (from the top-left to the bottom-
    right corner) and zeros elsewhere.

    :param N: The number of rows of the identity matrix.
    :param M: The number of columns of the identity matrix, defaults to
        `None` meaning M == N.
    :param k: Diagonal offset, defaults to `0`.
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
    out = zeros(N, M, dtype=dtype, order=order)
    for idx in range(N):
        out[idx, idx + k] = 1
    return out


identity = array_function_dispatch(lambda n: eye(n, None))


@array_function_dispatch
def tri(
    N: int,
    M: None | int = None,
    k: int = 0,
    dtype: DTypeLike | _BaseDType = None,
    order: _OrderKACF = None,
) -> ndarray:
    """Generate a lower triangular matrix with ones below and on the
    main diagonal, and zeros elsewhere.

    The function creates a two-dimensional array with the specified
    dimensions. The elements below and on the main diagonal are set to
    1, while the elements above the diagonal are set to 0.

    :param N: The number of rows of the identity matrix.
    :param M: The number of columns of the identity matrix, defaults to
        `None` meaning M == N.
    :param k: Diagonal offset, defaults to `0`.
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
    out = zeros(N, M, dtype=dtype, order=order)
    for idx in range(N):
        start = max(0, idx + k)
        end = min(M, idx + 1 + k)
        if start < end:
            out[idx, start:end] = 1
    return out


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
        size = len(v)
        nrows = size + max(0, k)
        ncols = size + max(0, -k)
        out = zeros(nrows, ncols, dtype=v.dtype)
        for idx in range(size):
            row, col = (idx + k, idx) if k < 0 else (idx, idx + k)
            out[row, col] = v[idx]
        return out
    elif v.ndim == 2:
        rows, cols = v.shape
        if k >= 0:
            start = max(0, k)
            end = min(rows, cols + k)
            size = max(0, end - start)
            return array([v[dim, dim + k] for dim in range(size)])
        else:
            start = max(0, -k)
            end = min(rows + k, cols)
            size = max(0, end - start)
            return array([v[dim - k, dim] for dim in range(size)])
    else:
        raise ValueError("Input array must be 1D or 2D")


@array_function_dispatch
def arange(*args: t.Any, dtype: DTypeLike = None) -> ndarray:
    """Return evenly spaced values within a given range.

    The `arange` function generates a 1-D array containing evenly spaced
    values over a specified interval. The interval is defined by the
    `start`, `stop`, and `step` arguments. It mimics the behavior of
    Python's built-in `range` function but returns an `ndarray` instead.

    :param dtype: The desired data type of the output array, defaults to
        `None`, in which case the data type is inferred.
    :return: A 1-D array of evenly spaced values.
    """
    c_args = len(args)
    if c_args == 0:
        raise TypeError("arange() requires stop to be specified")
    elif c_args > 3:
        raise TypeError("Too many input arguments")
    start, stop, step = (
        (0, args[0], 1)
        if c_args == 1
        else (args[0], args[1], 1) if c_args == 2 else args
    )
    if step == 0:
        raise ValueError("Step size must not be zero")
    size = max(0, math.ceil((stop - start) / step))
    dtype = (
        "int64"
        if all(isinstance(idx, int) for idx in (start, stop, step))
        else "float64"
    )
    out = empty(size, dtype=dtype)
    out[:] = [start + idx * step for idx in range(size)]
    return out


@array_function_dispatch
def matmul(a: ndarray, b: ndarray) -> ndarray | int | float:
    """Compute the matrix multiplication of two arrays.

    For 1D arrays, this function returns the inner product of the
    vectors. For 2D arrays, it performs matrix multiplication.
    For higher-dimensional arrays, it computes the matrix multiplication
    product along the last axis of `a` and the second-to-last axis of
    `b`.

    :param a: First input array.
    :param b: Second input array.
    :return: The matrix multiplication product of `a` and `b`.
    :raises ValueError: If the shapes of `a` and `b` are not aligned for
        matrix multiplication computation.

    .. note::

        [1] The output's shape depends on the broadcasting rules and the
            alignment of axes during computation.
    """
    if isinstance(a, ndarray) and isinstance(b, ndarray):
        if a.ndim == 1 and b.ndim == 1:
            if a.shape[0] != b.shape[0]:
                raise ValueError(
                    "Shapes of 1D arrays must be the same for dot product"
                )
            return sum(a[idx] * b[idx] for idx in range(a.shape[0]))
        elif a.ndim == 2 and b.ndim == 2:
            if a.shape[1] != b.shape[0]:
                raise ValueError(
                    "Shapes are not aligned for matrix multiplication"
                )
            out = ndarray((a.shape[0], b.shape[1]), dtype=a.dtype)
            for idx in range(a.shape[0]):
                for jdx in range(b.shape[1]):
                    out[idx, jdx] = sum(
                        a[idx, kdx] * b[kdx, jdx] for kdx in range(a.shape[1])
                    )
            return out
        elif a.ndim > 2 or b.ndim > 2:
            raise ValueError("Higher-dimensional dot product is not supported")
        else:
            raise ValueError("Invalid shapes for dot product")
    else:
        raise ValueError("Input operands does not have enough dimensions")


@array_function_dispatch
def dot(a: ndarray, b: ndarray) -> ndarray | int | float:
    """Compute the dot product of two arrays.

    For 1D arrays, this function returns the inner product of the
    vectors. For 2D arrays, it performs matrix multiplication.
    For higher-dimensional arrays, it computes the dot product along the
    last axis of `a` and the second-to-last axis of `b`.

    :param a: First input array.
    :param b: Second input array.
    :return: The dot product of `a` and `b`.
    :raises ValueError: If the shapes of `a` and `b` are not aligned for
        dot product computation.

    .. note::

        [1] The output's shape depends on the broadcasting rules and the
            alignment of axes during computation.
    """
    try:
        return matmul(a, b)
    except ValueError:
        return a * b


@array_function_dispatch
def add(x1: ndarray, x2: ndarray) -> ndarray | int | float:
    """Add arguments element-wise.

    For 1D arrays, this function returns the inner sum or addition of
    the vectors. For 2D arrays, it performs addition element-wise.
    For higher-dimensional arrays, it computes the addition along the
    last axis of `x1` and the second-to-last axis of `x2`.

    :param x1: First input array.
    :param x2: Second input array.
    :return: The element-wise addition of `x1` and `x2`.
    :raises ValueError: If the shapes of `x1` and `x2` are not aligned
        for addition.

    .. note::

        [1] The output's shape depends on the broadcasting rules and the
            alignment of axes during computation.
    """
    try:
        if x1.ndim == 1 and x2.ndim == 1:
            if x1.shape[0] != x2.shape[0]:
                raise ValueError(
                    "Shapes of 1D arrays must be the same for addition"
                )
            return array([idx + jdx for idx, jdx in zip(x1.flat, x2.flat)])
        elif x1.ndim == 2 and x2.ndim == 2:
            if x1.shape[1] != x2.shape[0]:
                raise ValueError("Shapes are not aligned for addition")
            out = ndarray((x1.shape[0], x2.shape[1]), dtype=x1.dtype)
            for idx in range(x1.shape[0]):
                for jdx in range(x2.shape[1]):
                    arr = []
                    for kdx in range(x1.shape[1]):
                        arr.append(x1[idx, kdx] + x2[kdx, jdx])
                    out[idx, jdx] = arr
            return out
        elif x1.ndim > 2 or x2.ndim > 2:
            raise ValueError("Higher-dimensional addition is not supported")
        else:
            raise ValueError("Invalid shapes for addition")
    except (AttributeError, TypeError):
        return x1 + x2


@array_function_dispatch
def subtract(x1: ndarray, x2: ndarray) -> ndarray | int | float:
    """Subtract arguments element-wise.

    For 1D arrays, this function returns the inner subtraction of
    the vectors. For 2D arrays, it performs subtraction element-wise.
    For higher-dimensional arrays, it computes the subtraction along the
    last axis of `x1` and the second-to-last axis of `x2`.

    :param x1: First input array.
    :param x2: Second input array.
    :return: The element-wise subtraction of `x1` and `x2`.
    :raises ValueError: If the shapes of `x1` and `x2` are not aligned
        for subtraction.

    .. note::

        [1] The output's shape depends on the broadcasting rules and the
            alignment of axes during computation.
    """
    try:
        if x1.ndim == 1 and x2.ndim == 1:
            if x1.shape[0] != x2.shape[0]:
                raise ValueError(
                    "Shapes of 1D arrays must be the same for subtraction"
                )
            return array([idx - jdx for idx, jdx in zip(x1.flat, x2.flat)])
        elif x1.ndim == 2 and x2.ndim == 2:
            if x1.shape[1] != x2.shape[0]:
                raise ValueError("Shapes are not aligned for subtraction")
            out = ndarray((x1.shape[0], x2.shape[1]), dtype=x1.dtype)
            for idx in range(x1.shape[0]):
                for jdx in range(x2.shape[1]):
                    arr = []
                    for kdx in range(x1.shape[1]):
                        arr.append(x1[idx, kdx] - x2[kdx, jdx])
                    out[idx, jdx] = arr
            return out
        elif x1.ndim > 2 or x2.ndim > 2:
            raise ValueError("Higher-dimensional subtraction is not supported")
        else:
            raise ValueError("Invalid shapes for subtraction")
    except (AttributeError, TypeError):
        return x1 - x2


@array_function_dispatch
def multiply(x1: ndarray, x2: ndarray) -> ndarray | int | float:
    """Multiply arguments element-wise.

    For 1D arrays, this function returns the inner multiplication of
    the vectors. For 2D arrays, it performs multiplication element-wise.
    For higher-dimensional arrays, it computes the multiplication along
    the last axis of `x1` and the second-to-last axis of `x2`.

    :param x1: First input array.
    :param x2: Second input array.
    :return: The element-wise multiplication of `x1` and `x2`.
    :raises ValueError: If the shapes of `x1` and `x2` are not aligned
        for multiplication.

    .. note::

        [1] The output's shape depends on the broadcasting rules and the
            alignment of axes during computation.
    """
    try:
        if x1.ndim == 1 and x2.ndim == 1:
            if x1.shape[0] != x2.shape[0]:
                raise ValueError(
                    "Shapes of 1D arrays must be the same for multiplication"
                )
            return array([idx * jdx for idx, jdx in zip(x1.flat, x2.flat)])
        elif x1.ndim == 2 and x2.ndim == 2:
            if x1.shape[1] != x2.shape[0]:
                raise ValueError("Shapes are not aligned for multiplication")
            out = ndarray((x1.shape[0], x2.shape[1]), dtype=x1.dtype)
            for idx in range(x1.shape[0]):
                for jdx in range(x2.shape[1]):
                    arr = []
                    for kdx in range(x1.shape[1]):
                        arr.append(x1[idx, kdx] * x2[kdx, jdx])
                    out[idx, jdx] = arr
            return out
        elif x1.ndim > 2 or x2.ndim > 2:
            raise ValueError(
                "Higher-dimensional multiplication is not supported"
            )
        else:
            raise ValueError("Invalid shapes for multiplication")
    except (AttributeError, TypeError):
        return x1 * x2


@array_function_dispatch
def divide(x1: ndarray, x2: ndarray) -> ndarray | float:
    """Divide arguments element-wise.

    For 1D arrays, this function returns the inner division of
    the vectors. For 2D arrays, it performs division element-wise.
    For higher-dimensional arrays, it computes the division along the
    last axis of `x1` and the second-to-last axis of `x2`.

    :param x1: First input array.
    :param x2: Second input array.
    :return: The element-wise division of `x1` and `x2`.
    :raises ValueError: If the shapes of `x1` and `x2` are not aligned
        for division.

    .. note::

        [1] The output's shape depends on the broadcasting rules and the
            alignment of axes during computation.
    """
    try:
        if x1.ndim == 1 and x2.ndim == 1:
            if x1.shape[0] != x2.shape[0]:
                raise ValueError(
                    "Shapes of 1D arrays must be the same for division"
                )
            tmp = []
            for idx, jdx in zip(x1.flat, x2.flat):
                try:
                    tmp.append(idx / jdx)
                except ZeroDivisionError:
                    tmp.append(xp.inf)
            return array(tmp)
        elif x1.ndim == 2 and x2.ndim == 2:
            if x1.shape[1] != x2.shape[0]:
                raise ValueError("Shapes are not aligned for division")
            out = ndarray((x1.shape[0], x2.shape[1]), dtype=x1.dtype)
            for idx in range(x1.shape[0]):
                for jdx in range(x2.shape[1]):
                    arr = []
                    for kdx in range(x1.shape[1]):
                        try:
                            arr.append(x1[idx, kdx] / x2[kdx, jdx])
                        except ZeroDivisionError:
                            arr.append(xp.inf)
                    out[idx, jdx] = arr
            return out
        elif x1.ndim > 2 or x2.ndim > 2:
            raise ValueError("Higher-dimensional division is not supported")
        else:
            raise ValueError("Invalid shapes for division")
    except (AttributeError, TypeError):
        return x1 / x2


truedivide = divide


@array_function_dispatch
def power(x1: ndarray, x2: ndarray) -> ndarray | int | float:
    """First array elements raised to powers from second array,
    element-wise.

    For 1D arrays, this function returns the exponentiation of the
    vectors. For 2D arrays, it performs exponentiation element-wise.
    For higher-dimensional arrays, it computes the exponentiation along
    the last axis of `x1` and the second-to-last axis of `x2`.

    :param x1: First input array.
    :param x2: Second input array.
    :return: The element-wise exponentiation of `x1` and `x2`.
    :raises ValueError: If the shapes of `x1` and `x2` are not aligned
        for exponentiation.

    .. note::

        [1] The output's shape depends on the broadcasting rules and the
            alignment of axes during computation.
    """
    try:
        if x1.ndim == 1 and x2.ndim == 1:
            if x1.shape[0] != x2.shape[0]:
                raise ValueError(
                    "Shapes of 1D arrays must be the same for exponentiation"
                )
            return array([idx**jdx for idx, jdx in zip(x1.flat, x2.flat)])
        elif x1.ndim == 2 and x2.ndim == 2:
            if x1.shape[1] != x2.shape[0]:
                raise ValueError("Shapes are not aligned for exponentiation")
            out = ndarray((x1.shape[0], x2.shape[1]), dtype=x1.dtype)
            for idx in range(x1.shape[0]):
                for jdx in range(x2.shape[1]):
                    arr = []
                    for kdx in range(x1.shape[1]):
                        arr.append(x1[idx, kdx] ** x2[kdx, jdx])
                    out[idx, jdx] = arr
            return out
        elif x1.ndim > 2 or x2.ndim > 2:
            raise ValueError(
                "Higher-dimensional exponentiation is not supported"
            )
        else:
            raise ValueError("Invalid shapes for exponentiation")
    except (AttributeError, TypeError):
        return x1**x2
