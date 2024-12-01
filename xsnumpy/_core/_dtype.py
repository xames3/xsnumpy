"""\
xsNumPy DType Implementation
============================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 18 2024
Last updated on: Sunday, December 01 2024

This module provides the foundational implementation of the
`xsnumpy._core._dtype` system, enabling flexible and type-safe handling
of data types within the xsNumPy library. A robust and extensible data
type system is a critical component of any numerical computing library,
as it facilitates precise control over the representation, manipulation,
and conversion of data.

The module defines a set of core data types and their properties,
encapsulated using a lightweight and efficient structure. Each data type
is represented as a named tuple containing attributes that describe its
kind (e.g., integer, floating point, boolean), name, and size in memory.
These attributes are designed to align with common conventions found in
numerical computing and scientific programming, ensuring familiarity and
ease of use for developers.

Additionally, the module includes utilities for converting between
different representations of data types. The `_convert_dtype` function
allows seamless translation of type specifications between formats such
as `numpy` strings, `ctypes`, and short notations. This functionality
ensures interoperability with external libraries and underlying
system-level type representations, making it easier to integrate
xsNumPy into diverse computational workflows.
"""

from __future__ import annotations

import ctypes
import typing as t
from collections import namedtuple

_dtype = namedtuple("_dtype", "kind, name, itemsize")

_dtypes: tuple[_dtype, ...] = (
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


def _convert_dtype(
    dtype: None | t.Any,
    to: t.Literal["short", "numpy", "ctypes"] = "numpy",
) -> None | t.Any:
    """Convert a data type representation to a specified format.

    This utility function facilitates the conversion of data type
    definitions into one of the supported formats (`short`, `numpy`, or
    `ctypes`) used within the xsNumPy library. The conversion allows
    users to seamlessly translate type definitions between different
    contexts, enabling better interoperability and integration with
    other systems and libraries.

    :param dtype: The input data type to be converted. If `None`, the
        function returns `None` else, the input is expected to be
        convertible to a string and should match one of the predefined
        types in the `_dtypes` tuples.
    :param to: The target format for the data type conversion, defaults
        to `numpy`.
    :return: The converted data type in the requested format. If the
        input `dtype` is not found in the `_dtypes` tuple, the function
        returns the original input value as is.
    :raises ValueError: If the `to` parameter is not one of the accepted
        values.
    """
    if dtype is None:
        return dtype
    dtype = str(dtype)
    try:
        idx = {"short": 0, "numpy": 1, "ctypes": 2}[to]
    except KeyError:
        raise ValueError(f"Invalid conversion target: {to!r}")
    else:
        for _dtype in _dtypes:
            if dtype in _dtype:
                return _dtype[idx]
    return dtype
