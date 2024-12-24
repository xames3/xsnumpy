"""\
xsNumPy N-Dimensional Array
===========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 18 2024
Last updated on: Tuesday, December 24 2024

This module implements the core functionality of the `xsnumpy` package,
providing the foundational `ndarray` class, which serves as the building
block for numerical computation in the library. Designed with
flexibility, efficiency, and modularity in mind, the `ndarray` class aims
to replicate and innovate upon the core features of NumPy arrays while
emphasizing a Python-standard-library-only approach.

The `ndarray` class in this module is a multidimensional container that
supports efficient manipulation of numerical data. Its design is inspired
by NumPy, the gold standard for numerical computing in Python, and it
incorporates similar semantics to provide users with a familiar
interface. Additionally, the class introduces a Pythonic, educational
perspective, making it suitable for learning and experimentation with
array mechanics without relying on external libraries.

As of now, the module supports features such as::

    - Efficient storage and representation of n-dimensional data.
    - Flexible shape manipulation, including reshaping and broadcasting.
    - Element-wise operations, including arithmetic, logical, and
      comparison operations, via rich operator overloading.
    - Slicing and indexing support for intuitive data access.
    - Conversion utilities to export data to native Python types
      (e.g., lists).
    - Basic numerical operations such as dot product, summation, and
      element reduction.

The `ndarray` implementation draws inspiration from NumPy's architecture
but deliberately simplifies and reimagines certain aspects for
educational purposes and to meet the constraints of pure Python. By
eschewing C or Cython extensions, the `ndarray` class offers an
accessible implementation that emphasizes algorithmic clarity over raw
performance.

The `ndarray` class exposes a set of intuitive attributes like::

    - `shape`: A tuple indicating the array's dimensions.
    - `ndim`: The number of dimensions of the array.
    - `size`: The total number of elements in the array.
    - `dtype`: Specifies the data type of the array's elements.

In addition to the core `ndarray` functionality, this module introduces
several helper functions to aid in array manipulation and generation.
These functions are designed to complement the `ndarray` class and mimic
key functionality found in NumPy.

While this module implements many fundamental features of `ndarray`, it
does not aim to match NumPy's performance or breadth. Instead, the focus
is on clarity, usability, and modularity, providing a platform for
learning and experimentation.
"""

from __future__ import annotations

import builtins
import ctypes
import itertools
import re
import typing as t
from collections import namedtuple
from collections.abc import Iterable

from xsnumpy import array_function_dispatch
from xsnumpy._typing import DTypeLike
from xsnumpy._typing import _OrderKACF
from xsnumpy._typing import _ShapeLike
from xsnumpy._utils import calc_size
from xsnumpy._utils import calc_strides
from xsnumpy._utils import get_step_size
from xsnumpy._utils import set_module

__all__: list[str] = [
    "bool",
    "float",
    "float32",
    "float64",
    "int16",
    "int32",
    "int64",
    "int8",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
]


def _dtype_repr(self: _BaseDType) -> str:
    """Add repr method to dtype namedtuple."""
    return f"dtype({self.numpy!r})"


def _dtype_str(self: _BaseDType) -> str:
    """Add str method to dtype namedtuple."""
    return f"{self.numpy}"


_BaseDType = namedtuple("_BaseDType", "short, numpy, ctypes, value")
_BaseDType.__repr__ = _dtype_repr
_BaseDType.__str__ = _dtype_str

_supported_dtypes: tuple[_BaseDType, ...] = (
    (bool := _BaseDType("b1", "bool", ctypes.c_bool, False)),
    (int8 := _BaseDType("i1", "int8", ctypes.c_int8, 0)),
    (uint8 := _BaseDType("u1", "uint8", ctypes.c_uint8, 0)),
    (int16 := _BaseDType("i2", "int16", ctypes.c_int16, 0)),
    (uint16 := _BaseDType("u2", "uint16", ctypes.c_uint16, 0)),
    (int32 := _BaseDType("i4", "int32", ctypes.c_int32, 0)),
    (uint32 := _BaseDType("u4", "uint32", ctypes.c_uint32, 0)),
    (int64 := _BaseDType("i8", "int64", ctypes.c_int64, 0)),
    (uint64 := _BaseDType("u8", "uint64", ctypes.c_uint64, 0)),
    (float := _BaseDType("f4", "float", ctypes.c_float, 0.0)),
    (float64 := _BaseDType("f8", "float64", ctypes.c_double, 0.0)),
)
float32 = float

for dtype in _supported_dtypes:
    globals()[dtype] = dtype


def _convert_dtype(
    dtype: None | t.Any,
    to: t.Literal["short", "numpy", "ctypes"] = "numpy",
) -> t.Any | str:
    """Convert a data type representation to the desired format.

    This utility function converts a data type string or object into one
    of the supported representations: `short`, `numpy`, or `ctypes`.

    :param dtype: The input data type to be converted.
    :param to: The target format for the data type conversion, defaults
        to `numpy`.
    :return: The converted data type in the requested format. If the
        input `dtype` is not found in the `_supported_dtypes` tuple, the
        function returns the original input value as is.
    :raises ValueError: If the `to` parameter is not one of the accepted
        values.
    """
    if dtype is None:
        return getattr(float64, to)
    try:
        idx = {"short": 0, "numpy": 1, "ctypes": 2}[to]
    except KeyError:
        raise ValueError(f"Invalid conversion target: {to!r}")
    else:
        for _dtype in _supported_dtypes:
            if dtype in _dtype:
                return _dtype[idx]
    return getattr(dtype, to)


@set_module("xsnumpy")
@array_function_dispatch
class ndarray:
    """Simplified implementation of a multi-dimensional array.

    An array object represents a multidimensional, homogeneous
    collection or list of fixed-size items. An associated data-type
    property describes the format of each element in the array.

    This class models a lightweight version of NumPy's `ndarray` for
    educational purposes, focusing on core concepts like shape, dtype,
    strides, and memory management in pure Python. It provides a
    foundational understanding of how array-like structures manage data
    and metadata.

    :param shape: The desired shape of the array. Can be an int for
        1D arrays or a sequence of ints for multidimensional arrays.
    :param dtype: The desired data type of the array, defaults to
        `xs.float32` if not specified.
    :param buffer: Object used to fill the array with data, defaults to
        `None`.
    :param offset: Offset of array data in buffer, defaults to `0`.
    :param strides: Strides of data in memory, defaults to `None`.
    :param order: The memory layout of the array, defaults to `None`.
    """

    __slots__: tuple[str, ...] = (
        "_base",
        "_data",
        "_dtype",
        "_itemsize",
        "_offset",
        "_shape",
        "_strides",
    )

    def __init__(
        self,
        shape: _ShapeLike,
        dtype: None | DTypeLike | _BaseDType = float32,
        buffer: None | t.Any = None,
        offset: t.SupportsIndex = 0,
        strides: None | _ShapeLike = None,
        order: None | _OrderKACF = None,
    ) -> None:
        """Initialize an `ndarray` object from the provided shape."""
        if order is not None:
            raise RuntimeError(
                f"{type(self).__name__} supports only C-order arrays; 'order'"
                " must be None"
            )
        if not isinstance(shape, Iterable):
            raise TypeError("Shape must be either tuple or list of integers")
        self._shape = tuple(int(dim) for dim in shape)
        if dtype is None:
            dtype = float32
        else:
            dtype = globals()[dtype]
        self._dtype = dtype
        self._itemsize = int(_convert_dtype(dtype, "short")[-1])
        self._offset = int(offset)
        if buffer is None:
            self._base = None
            if self._offset != 0:
                raise ValueError("Offset must be 0 when buffer is None")
            if strides is not None:
                raise ValueError("Strides must be None when buffer is None")
            self._strides = calc_strides(self._shape, self.itemsize)
        else:
            if isinstance(buffer, ndarray) and buffer.base is not None:
                buffer = buffer.base
            self._base = buffer
            if isinstance(buffer, ndarray):
                buffer = buffer.data
            if self._offset < 0:
                raise ValueError("Offset must be non-negative")
            if strides is None:
                strides = calc_strides(self._shape, self.itemsize)
            elif not (
                isinstance(strides, tuple)
                and all(isinstance(stride, int) for stride in strides)
                and len(strides) == len(self._shape)
            ):
                raise ValueError("Invalid strides provided")
            self._strides = tuple(strides)
        buffersize = self._strides[0] * self._shape[0] // self._itemsize
        buffersize += self._offset
        Buffer = _convert_dtype(dtype, "ctypes") * buffersize
        if buffer is None:
            if not isinstance(Buffer, str):
                self._data = Buffer()
        elif isinstance(buffer, ctypes.Array):
            self._data = Buffer.from_address(ctypes.addressof(buffer))
        else:
            self._data = Buffer.from_buffer(buffer)

    def _format_repr_as_str(
        self,
        s: str,
        axis: int,
        offset: int,
        pad: int = 0,
        whitespace: int = 0,
    ) -> str:
        """Format repr to mimic NumPy's ndarray as close as possible."""
        indent = min(2, max(0, (self.ndim - axis - 1)))
        if axis < len(self.shape):
            s += "["
            for idx, val in enumerate(range(self.shape[axis])):
                if idx > 0:
                    s += ("\n " + " " * pad + " " * axis) * indent
                _oset = offset + val * self._strides[axis] // self.itemsize
                s = self._format_repr_as_str(
                    s, axis + 1, _oset, whitespace=whitespace
                )
                if idx < self.shape[axis] - 1:
                    s += ", "
            s += "]"
        else:
            r = repr(self.data[offset])
            if "." in r and r.endswith(".0"):
                r = f"{r[:-1]:<{whitespace}}"
            else:
                r = f"{r:>{whitespace}}"
            s += r
        return s

    def __repr__(self) -> str:
        """Return a string representation of ndarray object."""
        ws = max(
            map(
                len,
                map(
                    str,
                    (self.data[_] for _ in range(self.size)),
                ),
            )
        )
        s = self._format_repr_as_str("", 0, self._offset, 6, ws)
        if (
            self.dtype != float64
            and self.dtype != int64
            and self.dtype != bool
        ):
            return f"array({s}, dtype={self.dtype.__str__()})"
        else:
            return f"array({s})"

    def __str__(self) -> str:
        """Return a printable representation of ndarray object."""
        sanitized = "".join(re.findall(r"\[\[.*?\]\]", repr(self), re.DOTALL))
        return sanitized.replace(",", "").replace(" " * 6, "")

    def __float__(self) -> None | builtins.float:
        """Convert the ndarray to a scalar float if it has exactly one
        element.

        This method attempts to convert an ndarray instance to a scalar
        float. The conversion is only possible if the ndarray contains
        exactly one element.
        """
        if self.size == 1:
            return builtins.float(self.data[self._offset])
        else:
            raise TypeError("Only arrays of size 1 can be converted to scalar")

    def __int__(self) -> None | int:
        """Convert the ndarray to a scalar int if it has exactly one
        element.

        This method attempts to convert an ndarray instance to a scalar
        int. The conversion is only possible if the ndarray contains
        exactly one element.
        """
        if self.size == 1:
            return int(self.data[self._offset])
        else:
            raise TypeError("Only arrays of size 1 can be converted to scalar")

    def __len__(self) -> int:
        """Return the size of the first dimension of the array.

        This implements the behavior of `len()` for the array object,
        providing the number of elements in the first axis.

        :return: Size of the first dimension.
        :raises IndexError: If the array has no dimensions.
        """
        if not self.shape:
            raise IndexError("Array has no dimensions")
        return self.shape[0]

    def __getitem__(
        self, key: int | slice | tuple[int | slice | None, ...]
    ) -> t.Any | "ndarray":
        """Retrieve an item or sub-array from the array.

        Supports both integer indexing and slicing. If the resulting
        selection is a scalar, returns the value. Otherwise, it returns
        a new `ndarray` view into the data.

        :param key: Index or slice object, or tuple of them.
        :return: Scalar or sub-array as per the indexing operation.
        :raises IndexError: For invalid indexing.
        :raises TypeError: For unsupported key types.
        """
        offset, shape, strides = self._calculate_offset_and_strides(key)
        if not shape:
            return self._data[offset]
        return ndarray(
            shape,
            self._dtype,
            buffer=self,
            offset=offset,
            strides=strides,
        )

    def __setitem__(
        self,
        key: int | slice | tuple[None | int | slice, ...],
        value: (
            builtins.float | int | t.Sequence[int | builtins.float] | ndarray
        ),
    ) -> None:
        """Set the value of an element or a slice in the array.

        The method supports getting individual elements, slices, and
        subarrays of an `ndarray`. It calculates the correct offset and
        strides based on the provided index/key and updates the
        underlying data buffer with the provided value.

        :param key: Index or slice to identify the element or subarray
            to update.
        :param value: The value to assign to the selected element or
            subarray.
        :raises ValueError: If the number of element in the value does
            not match the size of selected subarray.

        .. note::

            The value can be a single scalar (float or int), a list, or a
            tuple, but must match the shape and size of the subarray
            being updated.
        """
        offset, shape, strides = self._calculate_offset_and_strides(key)
        # NOTE(xames3): If shape is empty, update the value directly in
        # the data buffer.
        if not shape:
            self._data[offset] = value
            return
        view = ndarray(
            shape,
            self._dtype,
            buffer=self,
            offset=offset,
            strides=strides,
        )
        if isinstance(value, (builtins.float, int)):
            values = [value] * view.size
        elif isinstance(value, (tuple, list)):
            values = list(value)
        else:
            # TODO(xames3): Although this is fixed and properly
            # implemented, I need to make it mypy compliant and remove
            # the disabling comment.
            if not isinstance(value, ndarray):
                value = ndarray(value)  # type: ignore
            values = value._flat()
        if view.size != len(values):
            raise ValueError(
                "Number of elements in the value doesn't match the shape"
            )
        subviews = [view]
        idx = 0
        while subviews:
            subview = subviews.pop(0)
            if step_size := get_step_size(subview):
                block = values[idx : idx + subview.size]
                subview._data[
                    slice(
                        subview._offset,
                        subview._offset + subview.size * step_size,
                        step_size,
                    )
                ] = block
                idx += subview.size
            else:
                for dim in range(subview.shape[0]):
                    subviews.append(subview[dim])
        assert idx == len(values)

    def __add__(self, other: ndarray | int | builtins.float) -> ndarray:
        """Perform element-wise addition of the ndarray with a scalar or
        another ndarray.

        This method supports addition with scalars (int or float) and
        other ndarrays of the same shape. The resulting array is of the
        same shape and dtype as the input.

        :param other: The operand for addition. Can be a scalar or an
            ndarray of the same shape.
        :return: A new ndarray containing the result of the element-wise
            addition.
        :raises TypeError: If `other` is neither a scalar nor an
            ndarray.
        :raises ValueError: If `other` is an ndarray but its shape
            doesn't match `self.shape`.
        """
        if isinstance(other, int):
            out = ndarray(self.shape, self.dtype)
            out[:] = [x + other for x in self._data]
            return out
        elif isinstance(other, builtins.float):
            out = ndarray(self.shape, float32)
            out[:] = [x + other for x in self._data]
            return out
        elif isinstance(other, ndarray):
            dtype = (
                int32
                if all(
                    map(
                        lambda x: isinstance(x, int), (self.dtype, other.dtype)
                    )
                )
                else float32
            )
            out = ndarray(self.shape, dtype)
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            out[:] = [x + y for x, y in zip(self.flat, other.flat)]
            return out
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

    def __radd__(self, other: ndarray | int | builtins.float) -> ndarray:
        """Perform reverse addition, delegating to `__add__`.

        :param other: The left-hand operand.
        :return: The result of the addition.
        """
        return self.__add__(other)

    def __sub__(self, other: ndarray | int | builtins.float) -> ndarray:
        """Perform element-wise subtraction of the ndarray with a scalar
        or another ndarray.

        This method supports subtraction with scalars (int or float) and
        other ndarrays of the same shape. The resulting array is of the
        same shape and dtype as the input.

        :param other: The operand for subtraction. Can be a scalar or an
            ndarray of the same shape.
        :return: A new ndarray containing the result of the element-wise
            subtraction.
        :raises TypeError: If `other` is neither a scalar nor an
            ndarray.
        :raises ValueError: If `other` is an ndarray but its shape
            doesn't match `self.shape`.
        """
        if isinstance(other, int):
            out = ndarray(self.shape, self.dtype)
            out[:] = [x - other for x in self._data]
            return out
        elif isinstance(other, builtins.float):
            out = ndarray(self.shape, float32)
            out[:] = [x - other for x in self._data]
            return out
        elif isinstance(other, ndarray):
            dtype = (
                int32
                if all(
                    map(
                        lambda x: isinstance(x, int), (self.dtype, other.dtype)
                    )
                )
                else float32
            )
            out = ndarray(self.shape, dtype)
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            out[:] = [x - y for x, y in zip(self.flat, other.flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for -: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return out

    def __rsub__(self, other: ndarray | int | builtins.float) -> ndarray:
        """Perform reverse subtraction, delegating to `__sub__`.

        :param other: The left-hand operand.
        :return: The result of the subtraction.
        """
        return self.__sub__(other)

    def __mul__(self, other: ndarray | int | builtins.float) -> ndarray:
        """Perform element-wise multiplication of the ndarray with a
        scalar or another ndarray.

        This method supports multiplication with scalars (int or float)
        and other ndarrays of the same shape. The resulting array is of
        the same shape and dtype as the input.

        :param other: The operand for multiplication. Can be a scalar or
            an ndarray of the same shape.
        :return: A new ndarray containing the result of the element-wise
            multiplication.
        :raises TypeError: If `other` is neither a scalar nor an
            ndarray.
        :raises ValueError: If `other` is an ndarray but its shape
            doesn't match `self.shape`.
        """
        if isinstance(other, int):
            out = ndarray(self.shape, self.dtype)
            out[:] = [x * other for x in self._data]
            return out
        elif isinstance(other, builtins.float):
            out = ndarray(self.shape, float32)
            out[:] = [x * other for x in self._data]
            return out
        elif isinstance(other, ndarray):
            dtype = (
                int32
                if all(
                    map(
                        lambda x: isinstance(x, int), (self.dtype, other.dtype)
                    )
                )
                else float32
            )
            out = ndarray(self.shape, dtype)
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            out[:] = [x * y for x, y in zip(self.flat, other.flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return out

    def __rmul__(self, other: ndarray | int | builtins.float) -> ndarray:
        """Perform reverse multiplication, delegating to `__mul__`.

        :param other: The left-hand operand.
        :return: The result of the multiplication.
        """
        return self.__mul__(other)

    def __truediv__(self, other: ndarray | int | builtins.float) -> ndarray:
        """Perform element-wise division of the ndarray with a scalar or
        another ndarray.

        This method supports division with scalars (int or float) and
        other ndarrays of the same shape. The resulting array is of the
        same shape and dtype as the input.

        :param other: The operand for division. Can be a scalar or an
            ndarray of the same shape.
        :return: A new ndarray containing the result of the element-wise
            division.
        :raises TypeError: If `other` is neither a scalar nor an
            ndarray.
        :raises ValueError: If `other` is an ndarray but its shape
            doesn't match `self.shape`.
        """
        if other == 0:
            raise ZeroDivisionError
        if isinstance(other, int):
            out = ndarray(self.shape, self.dtype)
            out[:] = [x / other for x in self._data]
            return out
        elif isinstance(other, builtins.float):
            out = ndarray(self.shape, float32)
            out[:] = [x / other for x in self._data]
            return out
        elif isinstance(other, ndarray):
            dtype = (
                int32
                if all(
                    map(
                        lambda x: isinstance(x, int), (self.dtype, other.dtype)
                    )
                )
                else float32
            )
            out = ndarray(self.shape, dtype)
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            out[:] = [x / y for x, y in zip(self.flat, other.flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for /: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return out

    def __floordiv__(self, other: ndarray | int | builtins.float) -> ndarray:
        """Perform element-wise floor division of the ndarray with a
        scalar or another ndarray.

        This method supports division with scalars (int or float) and
        other ndarrays of the same shape. The resulting array is of the
        same shape and dtype as the input.

        :param other: The operand for division. Can be a scalar or an
            ndarray of the same shape.
        :return: A new ndarray containing the result of the element-wise
            division.
        :raises TypeError: If `other` is neither a scalar nor an
            ndarray.
        :raises ValueError: If `other` is an ndarray but its shape
            doesn't match `self.shape`.
        """
        if other == 0:
            raise ZeroDivisionError
        if isinstance(other, int):
            out = ndarray(self.shape, self.dtype)
            out[:] = [x // other for x in self._data]
            return out
        elif isinstance(other, builtins.float):
            out = ndarray(self.shape, float32)
            out[:] = [x // other for x in self._data]
            return out
        elif isinstance(other, ndarray):
            dtype = (
                int32
                if all(
                    map(
                        lambda x: isinstance(x, int), (self.dtype, other.dtype)
                    )
                )
                else float32
            )
            out = ndarray(self.shape, dtype)
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            out[:] = [x // y for x, y in zip(self.flat, other.flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for //: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return out

    def __lt__(self, other: ndarray | int | builtins.float) -> ndarray:
        """Perform element-wise less-than operation of the ndarray with
        a scalar or another ndarray.

        This method supports comparison with scalars (int or float) and
        other ndarrays of the same shape. The resulting array is of the
        same shape and dtype as the input.

        :param other: The operand for comparison. Can be a scalar or an
            ndarray of the same shape.
        :return: A new boolean ndarray containing the result of the
            element-wise less-than comparison.
        :raises TypeError: If `other` is neither a scalar nor an
            ndarray.
        :raises ValueError: If `other` is an ndarray but its shape
            doesn't match `self.shape`."""
        out = ndarray(self.shape, bool)
        if isinstance(other, (int, builtins.float)):
            out[:] = [x < other for x in self._data]
            return out
        elif isinstance(other, ndarray):
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            out[:] = [x < y for x, y in zip(self.flat, other.flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for <: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return out

    def __gt__(self, other: ndarray | int | builtins.float) -> ndarray:
        """Perform element-wise greater-than operation of the ndarray
        with a scalar or another ndarray.

        This method supports comparison with scalars (int or float) and
        other ndarrays of the same shape. The resulting array is of the
        same shape and dtype as the input.

        :param other: The operand for comparison. Can be a scalar or an
            ndarray of the same shape.
        :return: A new boolean ndarray containing the result of the
            element-wise greater-than comparison.
        :raises TypeError: If `other` is neither a scalar nor an
            ndarray.
        :raises ValueError: If `other` is an ndarray but its shape
            doesn't match `self.shape`."""
        out = ndarray(self.shape, bool)
        if isinstance(other, (int, builtins.float)):
            out[:] = [x > other for x in self._data]
            return out
        elif isinstance(other, ndarray):
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            out[:] = [x > y for x, y in zip(self.flat, other.flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for <: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return out

    def __le__(self, other: ndarray | int | builtins.float) -> ndarray:
        """Perform element-wise less-than-equal operation of the ndarray
        with a scalar or another ndarray.

        This method supports comparison with scalars (int or float) and
        other ndarrays of the same shape. The resulting array is of the
        same shape and dtype as the input.

        :param other: The operand for comparison. Can be a scalar or an
            ndarray of the same shape.
        :return: A new boolean ndarray containing the result of the
            element-wise less-than-equal comparison.
        :raises TypeError: If `other` is neither a scalar nor an
            ndarray.
        :raises ValueError: If `other` is an ndarray but its shape
            doesn't match `self.shape`."""
        out = ndarray(self.shape, bool)
        if isinstance(other, (int, builtins.float)):
            out[:] = [x <= other for x in self._data]
            return out
        elif isinstance(other, ndarray):
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            out[:] = [x <= y for x, y in zip(self.flat, other.flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for <: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return out

    def __ge__(self, other: ndarray | int | builtins.float) -> ndarray:
        """Perform element-wise greater-than-equal operation of the
        ndarray with a scalar or another ndarray.

        This method supports comparison with scalars (int or float) and
        other ndarrays of the same shape. The resulting array is of the
        same shape and dtype as the input.

        :param other: The operand for comparison. Can be a scalar or an
            ndarray of the same shape.
        :return: A new boolean ndarray containing the result of the
            element-wise greater-than-equal comparison.
        :raises TypeError: If `other` is neither a scalar nor an
            ndarray.
        :raises ValueError: If `other` is an ndarray but its shape
            doesn't match `self.shape`."""
        out = ndarray(self.shape, bool)
        if isinstance(other, (int, builtins.float)):
            out[:] = [x >= other for x in self._data]
            return out
        elif isinstance(other, ndarray):
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            out[:] = [x >= y for x, y in zip(self.flat, other.flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for <: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return out

    def _calculate_offset_and_strides(
        self, key: int | slice | tuple[None | int | slice, ...]
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        """Calculate offset, shape, and strides for an indexing
        operation.

        This helper method computes the array metadata required for
        retrieving a sub-array or value based on the provided key.
        It handles integers, slices, `Ellipsis`, and `None` indexing.

        :param key: Indexing specification (int, slice, tuple, etc.).
        :return: Tuple of (offset, shape, strides).
        :raises IndexError: For invalid axis indexing or bounds errors.
        :raises TypeError: For unsupported key types.
        """
        axis: int = 0
        offset: int = self._offset
        shape: list[int] = []
        strides: list[int] = []
        if not isinstance(key, tuple):
            key = (key,)
        for dim in key:
            if axis >= len(self._shape):
                raise IndexError("Too many indices for array")
            axissize = self._shape[axis]
            if isinstance(dim, int):
                if not (-axissize <= dim < axissize):
                    raise IndexError(
                        f"Index {dim} out of bounds for axis {axis}"
                    )
                dim = dim + axissize if dim < 0 else dim
                offset += dim * self._strides[axis] // self.itemsize
                axis += 1
            elif isinstance(dim, slice):
                start, stop, step = dim.indices(axissize)
                shape.append(-(-(stop - start) // step))
                strides.append(step * self._strides[axis])
                offset += start * self._strides[axis] // self.itemsize
                axis += 1
            elif dim is Ellipsis:
                raise TypeError("Ellipsis is not supported")
            elif dim is None:
                shape.append(1)
                stride = 1
                for _stride in self._strides[axis:]:
                    stride *= _stride
                strides.append(stride)
            else:
                raise TypeError(f"Invalid index type: {type(dim).__name__!r}")
        shape.extend(self.shape[axis:])
        strides.extend(self._strides[axis:])
        return offset, tuple(shape), tuple(strides)

    def _flat(self) -> list[int | builtins.float]:
        """Flatten the ndarray and return all its elements in a list.

        This method traverses through the ndarray and collects its
        elements into a single list, regardless of its shape or
        dimensionality. It handles contiguous memory layouts and
        non-contiguous slices, ensuring that all elements of the ndarray
        are included in the returned list.

        :return: A list containing all elements in the ndarray.
        """
        values: list[int | builtins.float] = []
        subviews = [self]
        while subviews:
            subview = subviews.pop(0)
            step_size = get_step_size(subview)
            if step_size:
                values += self._data[
                    slice(
                        subview._offset,
                        subview._offset + subview.size * step_size,
                        step_size,
                    )
                ]
            else:
                for dim in range(subview.shape[0]):
                    subviews.append(subview[dim])
        return values

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape of the array."""
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        """Return the strides for traversing the array dimensions."""
        return self._strides

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the array."""
        return len(self._shape)

    @property
    def data(self) -> t.Any:
        """Return the memory buffer holding the array elements."""
        return self._data

    @property
    def size(self) -> int:
        """Return total number of elements in an array."""
        return calc_size(self._shape)

    @property
    def itemsize(self) -> int:
        """Return the size, in bytes, of each array element."""
        return self._itemsize

    @property
    def nbytes(self) -> int:
        """Return number of byte size of an array."""
        return self.size * self.itemsize

    @property
    def base(self) -> None | t.Any:
        """Return underlying buffer (if any)."""
        return self._base

    @property
    def dtype(self) -> t.Any | str:
        """Return the data type of the array elements (mainly str)."""
        return self._dtype

    @property
    def flat(self) -> t.Generator[int | builtins.float]:
        """Flatten the ndarray and yield its elements one by one.

        This property allows you to iterate over all elements in the
        ndarray, regardless of its shape or dimensionality, in a
        flattened order. It yields the elements one by one, similar to
        Python's built-in `iter()` function, and handles both contiguous
        and non-contiguous memory layouts.

        :yield: The elements of the ndarray in row-major (C-style)
            order.

        .. note::

            [1] If the ndarray has non-contiguous strides, the method
                correctly handles the retrieval of elements using the
                appropriate step size.
            [2] The method uses a generator to yield elements lazily,
                which can be more memory-efficient for large arrays.
        """
        subviews = [self]
        while subviews:
            subview = subviews.pop(0)
            step_size = get_step_size(subview)
            if step_size:
                for dim in self._data[
                    slice(
                        subview._offset,
                        subview._offset + subview.size * step_size,
                        step_size,
                    )
                ]:
                    yield dim
            else:
                for dim in range(subview.shape[0]):
                    subviews.append(subview[dim])

    def all(self, axis: None | int = None) -> builtins.bool | ndarray:
        """Return True if all elements evaluate to True."""
        if axis is None:
            return all(self.flat)
        if not (0 <= axis < self.ndim):
            raise ValueError(
                f"Axis {axis} is out of bounds for array with "
                f"{self.ndim} dimensions"
            )
        shape = tuple(dim for idx, dim in enumerate(self.shape) if idx != axis)
        out = ndarray(shape, dtype=bool)
        indices = [slice(None)] * self.ndim
        for idx in ndindex(*shape):
            indices = list(idx[:axis]) + [slice(None)] + list(idx[axis:])
            out[idx] = all(element for element in self[tuple(indices)])
        return out

    def any(self, axis: None | int = None) -> builtins.bool | ndarray:
        """Return True if any elements evaluate to True."""
        if axis is None:
            return any(self.flat)
        if not (0 <= axis < self.ndim):
            raise ValueError(
                f"Axis {axis} is out of bounds for array with "
                f"{self.ndim} dimensions"
            )
        shape = tuple(dim for idx, dim in enumerate(self.shape) if idx != axis)
        out = ndarray(shape, dtype=bool)
        indices = [slice(None)] * self.ndim
        for idx in ndindex(*shape):
            indices = list(idx[:axis]) + [slice(None)] + list(idx[axis:])
            out[idx] = any(element for element in self[tuple(indices)])
        return out

    def astype(self, dtype: DTypeLike) -> ndarray:
        """Return a copy of the array cast to a specified data type.

        This method creates a new `ndarray` with the same shape and data
        as the original array but cast to the specified data type. The
        original array remains unmodified.

        :param dtype: The desired data type for the output array.
        :return: A new array with the specified data type and the same
            shape as the original array.
        :raises ValueError: If `dtype` is invalid or cannot be applied
            to the array.

        .. note::

            [1] This operation creates a copy of the data, even if the
                requested data type is the same as the original.
        """
        out = ndarray(self.shape, dtype)
        out[:] = self
        return out

    def clip(
        self,
        a_min: builtins.float | int | ndarray,
        a_max: builtins.float | int | ndarray,
        out: None | ndarray = None,
    ) -> ndarray:
        """Clip (limit) the values in the array.

        Given an input array, this method returns an array where values
        are limited to a specified range. All values less than `a_min`
        are set to `a_min`, and all values greater than `a_max` are set
        to `a_max`.

        :param a_min: Minimum value to clip to. Can be a scalar or an
            array.
        :param a_max: Maximum value to clip to. Can be a scalar or an
            array.
        :param out: Optional output array to store the result, defaults
            to `None`.
        :return: A new array with values clipped to the specified range.
        :raises TypeError: If either `a_min` or `a_max` are not either
            of type `int`, `float` or `ndarray`.
        :raises ValueError: If output shape doesn't match as the input
            array.
        """
        if not isinstance(a_min, (int, builtins.float, ndarray)):
            raise TypeError("`a_min` must be a scalar or an ndarray")
        if not isinstance(a_max, (int, builtins.float, ndarray)):
            raise TypeError("`a_max` must be a scalar or an ndarray")
        if isinstance(a_min, ndarray) and a_min.shape != self.shape:
            raise ValueError(
                "`a_min` must have the same shape as the input array"
            )
        if isinstance(a_max, ndarray) and a_max.shape != self.shape:
            raise ValueError(
                "`a_max` must have the same shape as the input array"
            )
        if out is None:
            out = ndarray(self.shape, self.dtype)
        F = self._flat()
        R = range(len(F))
        if isinstance(a_min, ndarray) and isinstance(a_max, ndarray):
            L = [min(a_max.flat[_], max(a_min.flat[_], F[_])) for _ in R]
        elif isinstance(a_min, ndarray):
            L = [min(a_max, max(a_min.flat[_], F[_])) for _ in R]
        elif isinstance(a_max, ndarray):
            L = [min(a_max.flat[_], max(a_min, F[_])) for _ in R]
        else:
            L = [min(a_max, max(a_min, _)) for _ in F]
        out[:] = L
        return out

    def copy(self) -> ndarray:
        """Return a deep copy of the array.

        This method creates a new `ndarray` instance with the same data,
        shape, and type as the original array. The copy is independent
        of the original, meaning changes to the copy do not affect the
        original array.

        :return: A new array with the same data, shape, and type as the
            original array.

        .. note::

            [1] This method ensures that both the data and metadata of
                the array are duplicated.
            [2] The `astype` method is used internally for copying,
                ensuring consistency and type fidelity.
        """
        return self.astype(self.dtype)

    def fill(self, value: int | builtins.float) -> None:
        """Fill the entire ndarray with a scalar value.

        This method assigns the given scalar value to all elements in
        the ndarray. The operation modifies the array in place and
        supports both integers and floating-point numbers as input.

        :param value: The scalar value to fill the ndarray with.
        :raises ValueError: If the provided `value` is not an integer
            or floating-point number.

        .. note::

            [1] This method modifies the ndarray in place.
            [2] The method uses slicing (`self[:] = value`) to
                efficiently set all elements to the specified value.
        """
        if not isinstance(value, (int, builtins.float)):
            raise ValueError("Value must be an integer or a float")
        self[:] = value

    def flatten(self, order: None | _OrderKACF = None) -> ndarray:
        """Return a copy of the array collapsed into one dimension."""
        if order is not None:
            raise ValueError("Order needs to be None")
        out = ndarray((self.size,), self.dtype)
        out[:] = self
        return out

    def item(self, args: None | int | tuple[int, int] = None) -> t.Any:
        """Return standard scalar Python object for ndarray object."""
        out = self.flatten()
        if args is None:
            if self.size == 1:
                return out[0]
            else:
                raise ValueError(
                    "Only array of size 1 can be converted to a Python scalar"
                )
        if isinstance(args, int):
            return out[args]
        elif isinstance(args, tuple):
            return self[args[0], args[1]]

    def tolist(self) -> list[t.Any]:
        """Convert the ndarray to a nested Python list.

        This method recursively iterates over the dimensions of the
        ndarray to construct a nested list that mirrors the shape and
        contents of the array.

        :return: A nested Python list representation of the ndarray's
            data.
        """
        # TODO(xames3): Write a flattening function to resolve nesting?
        raise NotImplementedError

    def view(
        self,
        dtype: None | DTypeLike = None,
        type: None | t.Any = None,
    ) -> None | ndarray:
        """Create a new view of the ndarray with a specified data type.

        This method allows creating a new ndarray view with a specified
        `dtype`. If no `dtype` is provided, the existing dtype of the
        array is used. The method supports efficient reinterpretation of
        the data buffer and respects the shape and strides of the
        original array. For 1D arrays, the dtype can differ if the total
        number of bytes remains consistent.

        :param dtype: The desired data type for the new view. If not
            provided, the current dtype is used.
        :param type: Ignored in this implementation.
        :return: A new ndarray view with the specified dtype. Returns
            `None` if the view cannot be created.
        :raises ValueError: If the array is multidimensional and the
            requested `dtype` differs from the current `dtype`.

        .. note::

            [1] For 1D arrays, changing the `dtype` adjusts the size
                based on the ratio of original item size to the new
                dtype's item size.
            [2] For multidimensional arrays, a new dtype must match the
                original dtype.
            [3] This method does not support modifying the `type`
                parameter, which is reserved for potential future
                extensions (most probably).
        """
        if dtype is None:
            dtype = self.dtype
        if dtype == self.dtype:
            return ndarray(
                self.shape,
                dtype,
                buffer=self,
                offset=self._offset,
                strides=self.strides,
            )
        elif self.ndim == 1:
            itemsize = int(_convert_dtype(dtype, "short")[-1])
            size = self.nbytes // itemsize
            offset = (self._offset * self.itemsize) // itemsize
            return ndarray((size,), dtype, buffer=self, offset=offset)
        else:
            raise ValueError("Arrays can only be viewed with the same dtype")


@set_module("xsnumpy")
@array_function_dispatch
class ndindex:
    """An iterator to generate all possible indices for a given shape.

    Similar to NumPy's `ndindex`, this class produces tuples
    representing the coordinates of elements in a multidimensional array
    with the specified shape.
    """

    def __init__(self, *shape: t.SupportsIndex) -> None:
        """Initialize the `ndindex` object with shape."""
        self.shape = shape

    def __iter__(self) -> t.Iterator[tuple[int, ...]]:
        """Return an iterator over all possible indices."""
        return itertools.product(*(range(dim) for dim in self.shape))
