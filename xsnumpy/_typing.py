"""\
xsNumPy DType Typing Implementation
===================================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 25 2024
Last updated on: Monday, December 02 2024

This module is a key component in the xsNumPy library, providing
essential type annotations and typing constructs to ensure a robust and
type-safe experience when working with the library. It leverages
Python's modern type hinting capabilities introduced through the
`typing` and `collections.abc` modules to define precise and flexible type
aliases used throughout the xsNumPy ecosystem.

The primary focus of this module is to define type constructs that enable
developers to work seamlessly with a wide variety of data types in
numerical computing, ensuring compatibility, clarity, and correctness in
function signatures and operations. These type definitions allow for
better code introspection, enable static type checking, and provide
clearer documentation to end users.

Additionally, this module incorporates advanced type features such as
union types, literal types, and nested type structures. These features
cater to the specific needs of numerical and scientific computing, where
arrays and data structures often contain deeply nested or structured
data. The design of this module is inspired by type annotations found in
modern numerical computing libraries, with careful consideration for
balancing expressiveness and simplicity. As part of the xsNumPy library,
it aims to bridge the gap between dynamic runtime behavior and static
type safety, providing a foundation for building reliable and
maintainable numerical computing applications.
"""

from __future__ import annotations

import typing as t
from collections.abc import Sequence

__all__: list[str] = [
    "DTypeLike",
    "_DTypeLikeNested",
    "_OrderKACF",
    "_Shape",
    "_ShapeLike",
    "_VoidDTypeLike",
]

_Shape: t.TypeAlias = tuple[int, ...]
_ShapeLike: t.TypeAlias = t.SupportsIndex | Sequence[t.SupportsIndex]
_OrderKACF: t.TypeAlias = t.Literal[None, "K", "A", "C", "F"]
_DTypeLikeNested: t.TypeAlias = t.Any
_VoidDTypeLike: t.TypeAlias = (
    tuple[_DTypeLikeNested, int]
    | tuple[_DTypeLikeNested, _ShapeLike]
    | list[t.Any]
    | tuple[_DTypeLikeNested, _DTypeLikeNested]
)
DTypeLike: t.TypeAlias = None | type[t.Any] | str | _VoidDTypeLike
