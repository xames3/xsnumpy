"""\
xsNumPy
=======

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, November 18 2024
Last updated on: Monday, November 18 2024

A personal pet-project of mine to try and implement the basic and bare-
bones functionality of NumPy just using pure Python.
"""

from __future__ import annotations

import xsnumpy._typing as typing

from ._core import *
from ._utils import *
from .version import version

__all__: list[str] = ["typing", "version"]
__all__ += _core.__all__
__all__ += _utils.__all__
