"""\
xsNumPy Array
==============

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 18 2024
Last updated on: Monday, November 25 2024

This module provides foundational structures and utilities for
array-like data structures modeled after NumPy's `ndarray`. It includes
support for core features like shapes, strides, and data type
definitions.
"""

from __future__ import annotations

import ctypes
import sys
import typing as t
from collections import namedtuple
from collections.abc import Iterable

if sys.version_info < (3, 12):
    from typing_extensions import Buffer as _SupportsBuffer
else:
    from collections.abc import Buffer as _SupportsBuffer

from ._typing import DTypeLike
from ._typing import _OrderKACF
from ._typing import _ShapeLike
from ._utils import calc_strides

__all__: list[str] = [
    "_common_types",
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

_dtype = namedtuple("_dtype", "short, numpy, ctypes")

_supported_dtypes: tuple[_dtype, ...] = (
    (bool := _dtype("b1", "bool", ctypes.c_bool)),
    (int8 := _dtype("i1", "int8", ctypes.c_int8)),
    (uint8 := _dtype("u1", "uint8", ctypes.c_uint8)),
    (int16 := _dtype("i2", "int16", ctypes.c_int16)),
    (uint16 := _dtype("u2", "uint16", ctypes.c_uint16)),
    (int32 := _dtype("i4", "int32", ctypes.c_int32)),
    (uint32 := _dtype("u4", "uint32", ctypes.c_uint32)),
    (int64 := _dtype("i8", "int64", ctypes.c_int64)),
    (uint64 := _dtype("u8", "uint64", ctypes.c_uint64)),
    (float := _dtype("f4", "float", ctypes.c_float)),
    (float64 := _dtype("f8", "float64", ctypes.c_double)),
)
_common_types: list[str] = [_dtype.numpy for _dtype in _supported_dtypes]
float32 = float


def _convert_dtype(
    dtype: None | t.Any,
    to: t.Literal["short", "numpy", "ctypes"] = "numpy",
) -> None | t.Any:
    """Convert a data type representation to the desired format.

    This utility function converts a data type string or object into one
    of the supported representations: `short`, `numpy`, or `ctypes`.

    :param dtype: The input data type to be converted. If `None`, the
        function returns `None` else, the input is expected to be
        convertible to a string and should match one of the predefined
        types in the `_supported_dtypes` tuple.
    :param to: The target format for the data type conversion, defaults
        to `numpy`.
    :return: The converted data type in the requested format. If the
        input `dtype` is not found in the `_supported_dtypes` tuple, the
        function returns the original input value as is.
    :raises ValueError: If the `to` parameter is not one of the accepted
        values.
    """
    if dtype is None:
        return None
    dtype = str(dtype)
    try:
        idx = {"short": 0, "numpy": 1, "ctypes": 2}[to]
    except KeyError:
        raise ValueError(f"Invalid conversion target: {to!r}")
    else:
        for _dtype in _supported_dtypes:
            if dtype in _dtype:
                return _dtype[idx]
    return dtype


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
        dtype: DTypeLike = "float64",
        buffer: None | _SupportsBuffer = None,
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
        dtype = _convert_dtype(dtype) if dtype is not None else "float64"
        if dtype not in _common_types:
            raise TypeError(f"Unsupported dtype: {dtype!r}")
        self._dtype = dtype
        self._itemsize = int(_convert_dtype(_dtype, "short")[-1])
        self._offset = int(offset)
        if buffer is None:
            self._base = None
            if self._offset != 0:
                raise ValueError("Offset must be 0 when buffer is None")
            if strides is not None:
                raise ValueError("Strides must be None when buffer is None")
            self._strides = calc_strides(self._shape, self._itemsize)
        else:
            self._base = buffer.base if isinstance(buffer, ndarray) else buffer
            self._data = buffer.data if isinstance(buffer, ndarray) else buffer
            if self._offset < 0:
                raise ValueError("Offset must be non-negative")
            if strides is None:
                strides = calc_strides(self._shape, self._itemsize)
            elif not (
                isinstance(strides, tuple)
                and all(isinstance(stride, int) for stride in strides)
                and len(strides) == len(self._shape)
            ):
                raise ValueError("Invalid strides provided")
            self._strides = tuple(strides)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the array."""
        return len(self._shape)

    @property
    def strides(self) -> tuple[int, ...]:
        """Return the strides for traversing the array dimensions."""
        return self._strides

    @property
    def dtype(self) -> str:
        """Return the data type of the array elements."""
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
