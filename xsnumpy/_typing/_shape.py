"""\
xsNumPy Shape Typing Implementation
===================================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 25 2024
Last updated on: Wednesday, November 27 2024

This module provides type definitions specifically designed to handle
shape  information in the xsNumPy library. Shapes are a fundamental
concept in numerical computing, describing the dimensions and structure
of arrays. This module defines precise and flexible type annotations to
represent shapes and shape-like constructs, ensuring consistent handling
of array dimensions across the library.

By leveraging modern Python type hinting capabilities, the module
introduces type aliases that cater to common use cases in numerical
operations. These types accommodate both strict representations of
shapes as tuples of integers and more flexible representations that
include index-like values from sequences. Such flexibility is crucial in
a library like xsNumPy, where inputs often vary in structure and detail.
"""

from __future__ import annotations

import typing as t
from collections.abc import Sequence

_Shape: t.TypeAlias = tuple[int, ...]
_ShapeLike: t.TypeAlias = t.SupportsIndex | Sequence[t.SupportsIndex]
