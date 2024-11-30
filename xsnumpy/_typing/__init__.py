"""\
xsNumPy Typing Entrypoint
=========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 25 2024
Last updated on: Wednesday, November 27 2024

This module serves as the core entry point for all type-related
constructs within the xsNumPy library, consolidating and exposing key
type definitions for consistent and type-safe usage across the library.
By aggregating and re-exporting type annotations from multiple internal
modules, this module provides a unified interface for handling data type
and shape specifications, which are fundamental to numerical and
array-based computations.

The primary goal of this module is to centralize type definitions,
enhancing maintainability and readability while minimizing redundancy
in type handling. It achieves this by importing and exposing relevant
type constructs defined in submodules, such as data type representations
(`_dtype_like`) and shape specifications (`_shape`). The `__all__`
sequence is constructed dynamically to ensure that only the intended
symbols are exported, preserving clarity and preventing namespace
pollution.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._dtype_like import DTypeLike
from ._dtype_like import _DTypeLikeNested
from ._dtype_like import _OrderKACF
from ._dtype_like import _VoidDTypeLike
from ._shape import _Shape
from ._shape import _ShapeLike

__all__: Sequence[str] = (
    "DTypeLike",
    "_DTypeLikeNested",
    "_OrderKACF",
    "_VoidDTypeLike",
    "_Shape",
    "_ShapeLike",
)
