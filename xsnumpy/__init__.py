"""\
xsNumPy
=======

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 18 2024
Last updated on: Saturday, January 04 2024

A personal pet-project of mine to try and implement the basic and bare-
bones functionality of NumPy just using pure Python. This module is a
testament to the richness of NumPy's design. By reimplementing its core
features in a self-contained and minimalistic fashion, this project
aims to::

    - Provide an educational tool for those seeking to understand array
      mechanics.
    - Serve as a lightweight alternative for environments where
      dependencies must be minimized.
    - Encourage developers to explore the intricacies of
      multidimensional array computation.

This `xsnumpy` project acknowledges the incredible contributions of the
NumPy team and community over decades of development. While this module
reimagines NumPy's functionality, it owes its design, inspiration, and
motivation to the pioneering work of the NumPy developers. This module
is not a replacement for NumPy but an homage to its brilliance and an
opportunity to explore its concepts from the ground up.
"""

from __future__ import annotations

import typing as t

import xsnumpy._typing as typing

from .version import version

__all__: list[str] = [
    "array_function_dispatch",
    "random",
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
__all__ += _utils.__all__

import xsnumpy._random as random


def info(object: t.Callable[..., _T]) -> str:
    """Show information about the xsNumPy objects."""
    return object.__doc__.replace("\n    ", "\n")
