"""\
xsNumPy Array
==============

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 18 2024
Last updated on: Friday, December 06 2024

This module provides foundational structures and utilities for
array-like data structures modeled after NumPy's `ndarray`. It includes
support for core features like shapes, strides, and data type
definitions.
"""

from __future__ import annotations

import builtins
import ctypes
import typing as t
from collections import namedtuple
from collections.abc import Iterable

from ._typing import DTypeLike
from ._typing import _OrderKACF
from ._typing import _ShapeLike
from ._utils import calc_size
from ._utils import calc_strides
from ._utils import get_step_size

__all__: list[str] = [
    "_convert_dtype",
    "bool",
    "float",
    "float32",
    "float64",
    "int16",
    "int32",
    "int64",
    "int8",
    "ndarray",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
]


def _dtype_repr(self: _base_dtype) -> str:
    """Add repr method to dtype namedtuple."""
    return f"xp.{self.numpy}({self.value})"


def _dtype_str(self: _base_dtype) -> str:
    """Add str method to dtype namedtuple."""
    return f"{self.numpy}"


_base_dtype = namedtuple("_base_dtype", "short, numpy, ctypes, value")
_base_dtype.__repr__ = _dtype_repr
_base_dtype.__str__ = _dtype_str

_supported_dtypes: tuple[_base_dtype, ...] = (
    (bool := _base_dtype("b1", "bool", ctypes.c_bool, False)),
    (int8 := _base_dtype("i1", "int8", ctypes.c_int8, 0)),
    (uint8 := _base_dtype("u1", "uint8", ctypes.c_uint8, 0)),
    (int16 := _base_dtype("i2", "int16", ctypes.c_int16, 0)),
    (uint16 := _base_dtype("u2", "uint16", ctypes.c_uint16, 0)),
    (int32 := _base_dtype("i4", "int32", ctypes.c_int32, 0)),
    (uint32 := _base_dtype("u4", "uint32", ctypes.c_uint32, 0)),
    (int64 := _base_dtype("i8", "int64", ctypes.c_int64, 0)),
    (uint64 := _base_dtype("u8", "uint64", ctypes.c_uint64, 0)),
    (float := _base_dtype("f4", "float", ctypes.c_float, 0.0)),
    (float64 := _base_dtype("f8", "float64", ctypes.c_double, 0.0)),
)
float32 = float


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
        dtype: DTypeLike | None = "float64",
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
        self._dtype = _convert_dtype(dtype)
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

    def __repr__(self) -> str:
        """Return a string representation of ndarray object."""

        def _extended_repr(s: str, axis: int, offset: int) -> str:
            indent = min(2, max(0, (self.ndim - axis - 1)))
            if axis < len(self.shape):
                s += "["
                for idx, val in enumerate(range(self.shape[axis])):
                    if idx > 0:
                        s += ("\n       " + " " * axis) * indent
                    _oset = offset + val * self._strides[axis] // self.itemsize
                    s = _extended_repr(s, axis + 1, _oset)
                    if idx < self.shape[axis] - 1:
                        s += ", "
                s += "]"
            else:
                r = repr(self.data[offset])
                if "." in r and r.endswith(".0"):
                    r = r[:-1]
                s += r
            return s

        s = _extended_repr("", 0, self._offset)
        if (
            self.dtype != "float64"
            and self.dtype != "int64"
            and self.dtype != "bool"
        ):
            return f"array({s}, dtype={self.dtype.__str__()})"
        else:
            return f"array({s})"

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
        key: int | slice | tuple[int | slice | None, ...],
        value: builtins.float | int | t.Sequence[int | builtins.float],
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
            # TODO(xames3): Need to come up with an alternative.
            ...
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

    def _calculate_offset_and_strides(
        self, key: int | slice | tuple[int | slice | None, ...]
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

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape of the array."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the array."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """Return total number of elements in an array."""
        return calc_size(self._shape)

    @property
    def nbytes(self) -> int:
        """Return number of byte size of an array."""
        return self.size * self.itemsize

    @property
    def strides(self) -> tuple[int, ...]:
        """Return the strides for traversing the array dimensions."""
        return self._strides

    @property
    def dtype(self) -> str:
        """Return the data type of the array elements (mainly str)."""
        return self._dtype

    @property
    def itemsize(self) -> int:
        """Return the size, in bytes, of each array element."""
        return self._itemsize

    @property
    def base(self) -> None | t.Any:
        """Return underlying buffer (if any)."""
        return self._base

    @property
    def data(self) -> t.Any:
        """Return the memory buffer holding the array elements."""
        return self._data
