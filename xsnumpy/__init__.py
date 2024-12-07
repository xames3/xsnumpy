"""\
xsNumPy
=======

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 18 2024
Last updated on: Saturday, December 07 2024

A personal pet-project of mine to try and implement the basic and bare-
bones functionality of NumPy just using pure Python.
"""

from __future__ import annotations

import typing as t

import xsnumpy._typing as typing

from .version import version

__all__: list[str] = [
    "array_function_dispatch",
    "typing",
    "version",
]

_T = t.TypeVar("_T")


def array_function_dispatch(func: t.Callable[..., _T]) -> t.Callable[..., _T]:
    """Decorator to register a function in the global namespace.

    This utility allows for automatic exposure of decorated functions to
    module-level imports. It ensures the function is added to both
    the global scope and the `__all__` list for proper namespace
    management.

    :param func: The function to be registered and exposed.
    :return: The original function, unmodified.
    """
    globals()[func.__name__] = func
    __all__.append(func.__name__)
    return func


from ._core import *
from ._numeric import *
from ._utils import *

__all__ += _core.__all__


def info(object: t.Callable[..., _T]) -> str:
    """Show information about the xsNumPy objects."""
    return object.__doc__.replace("\n    ", "\n")
