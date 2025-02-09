"""\
xsNumPy Utilities
=================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 25 2024
Last updated on: Wednesday, January 22 2025

This module provides utility functions designed to streamline and
enhance the development process within the xsNumPy library. These
utilities include helper functions that improve code organization,
modularity, and introspection. These utilities focus on enhancing code
modularity and enabling type-safe operations.
"""

from __future__ import annotations

import math
import typing as t

from xsnumpy import array_function_dispatch

if t.TYPE_CHECKING:
    from xsnumpy import ndarray
    from xsnumpy._typing import _ArrayType
    from xsnumpy._typing import _ShapeLike

__all__: list[str] = [
    "e",
    "inf",
    "nan",
    "newaxis",
    "pi",
]

e: float = math.e
inf: float = float("inf")
nan: float = float("nan")
newaxis: t.NoneType = None
pi: float = math.pi


@array_function_dispatch
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


@array_function_dispatch
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


@array_function_dispatch
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


@array_function_dispatch
def calc_shape_from_obj(object: t.Any) -> _ShapeLike:
    """Calculate the shape of a nested iterable object.

    This function recursively determines the dimensions of a nested
    structure, such as a list of lists, and returns its shape as a tuple
    of integers.

    :param object: The input object whose shape is to be determined. It
        can be a nested iterable or any other type.
    :return: A tuple representing the shape of the object.
    """
    shape: list[int] = []

    def _calc_shape(elements: t.Any, axis: int) -> None:
        """Helper function to calculate shape recursively."""
        if isinstance(elements, t.Sized) and not isinstance(
            elements, (str, bytes)
        ):
            if len(shape) <= axis:
                shape.append(0)
            current_len = len(elements)
            if current_len > shape[axis]:
                shape[axis] = current_len
            for element in elements:
                _calc_shape(element, axis + 1)

    _calc_shape(object, 0)
    return tuple(shape)


@array_function_dispatch
def has_uniform_shape(object: _ArrayType) -> bool:
    """Check if the input iterable has a uniform shape."""
    if not isinstance(object, t.Iterable):
        return True
    return (
        all(has_uniform_shape(element) for element in object)
        and len(
            set(
                len(element)
                for element in object
                if isinstance(element, t.Sized)
            )
        )
        <= 1
    )


@array_function_dispatch
def set_module(module: str) -> t.Callable[..., t.Any]:
    """Decorator for overriding `__module__` on a function or class."""

    def decorator(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
        """Inner function."""
        if module is not None:
            func.__module__ = module
        return func

    return decorator


@array_function_dispatch
def broadcast_shape(input: _ShapeLike, other: _ShapeLike) -> tuple[int, ...]:
    """Calculate the broadcast-compatible shape for two array.

    This function aligns the two shapes from the right, padding the
    smaller shape with `1`s on the left. Then, it checks compatibility
    for broadcasting::

        - Each array has at least one dimension.
        - Dimension sizes must either be equal, one of them is 1 or
          one of them does not exist.

    :param input: Shape of the first array.
    :param other: Shape of the second array.
    :return: The broadcast-compatible shape.
    :raises ValueError: If the shapes are incompatible for broadcasting.
    """
    buffer: list[int] = []
    r_input = list(reversed(input))
    r_other = list(reversed(other))
    maximum = max(len(r_input), len(r_other))
    r_input.extend([1] * (maximum - len(r_input)))
    r_other.extend([1] * (maximum - len(r_other)))
    for idx, jdx in zip(r_input, r_other):
        if idx == jdx or idx == 1 or jdx == 1:
            buffer.append(max(idx, jdx))
        else:
            raise ValueError(
                f"Operands couldn't broadcast together with shapes {input} "
                f"and {other}"
            )
    return tuple(reversed(buffer))


@array_function_dispatch
def normal_exp(value: float) -> float:
    """Dummy function to type safe compute exponentiations."""
    return math.exp(value)


@array_function_dispatch
def safe_exp(value: float) -> float:
    """Dummy function to type safe compute negative exponentiations."""
    return math.exp(-value)


@array_function_dispatch
def safe_max(arg1: float, arg2: float = 0.0) -> float:
    """Dummy function to type safe compute maximum values."""
    return max(arg1, arg2)


@array_function_dispatch
def safe_round(number: float, ndigits: int = 4) -> float:
    """Dummy function to type safe round floating values."""
    return round(number, ndigits)


@array_function_dispatch
def safe_range(args: t.Any) -> range:
    """Dummy function to type safe the range iterator."""
    if len(args) == 0:
        return range(0)
    elif len(args) == 1:
        return range(args[0])
    return range(args)
