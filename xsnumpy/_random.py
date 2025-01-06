"""\
xsNumPy Random
==============

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, January 04 2025
Last updated on: Monday, January 06 2025

This module implements pseudo-random number generators (PRNGs or RNGs)
with ability to draw samples from a variety of probability
distributions.
"""

from __future__ import annotations

import itertools
import random

from xsnumpy._core import ndarray
from xsnumpy._typing import DTypeLike
from xsnumpy._typing import _ShapeLike


class BitGenerator(random.Random):
    """Base class for Generator."""


class Generator:
    """A random number generator supporting multiple distributions.

    This generator serves as the backbone for random number generation.
    It can produce uniform and normal distributions.

    :param seed: Optional seed for reproducibility, defaults to `None`.
    """

    def __init__(self, bit_generator: BitGenerator) -> None:
        """Initialize the random generator with some seed."""
        self.generator = bit_generator

    def __repr__(self) -> str:
        """Return a string representation of Generator object."""
        return f"{type(self).__name__}(PCG64)"

    def random(
        self,
        size: None | int | _ShapeLike = None,
        dtype: None | DTypeLike = None,
    ) -> ndarray | float:
        """Return random floats in the half-open interval [0.0, 1.0).

        :param size: Size of the output, defaults to `None`.
        :param dtype: The desired data type of the output array, defaults to
            `None`.
        :return: A new array populated with random floating numbers from
            range [0.0, 1.0).
        """
        if size is None:
            return self.generator.random()
        if isinstance(size, int):
            out = ndarray((size,), dtype)
            out[:] = [self.generator.random() for _ in range(size)]
            return out
        elif isinstance(size, tuple):
            out = ndarray(size, dtype)
            N = range(max(size))
            for dim in itertools.product(N, N):
                try:
                    out[dim] = self.generator.random()
                except IndexError:
                    continue
            return out
        else:
            raise TypeError(
                f"Expected a sequence of integers or a single integer, "
                f"got {size!r}"
            )

    def integers(
        self,
        low: int,
        high: None | int = None,
        size: None | int | _ShapeLike = None,
    ) -> ndarray | int:
        """Return random integers.

        :param low: Lower bound (inclusive).
        :param high: Upper bound (exclusive), defaults to `None`.
        :param size: Size of the output, defaults to `None`.
        :return: A new array populated with random integers numbers from
            range [low, high].
        """
        if high is None:
            low, high = 0, low
        if size is None:
            return self.generator.randint(low, high - 1)
        if isinstance(size, int):
            out = ndarray((size,), int)
            out[:] = [
                self.generator.randint(low, high - 1) for _ in range(size)
            ]
            return out
        elif isinstance(size, tuple):
            out = ndarray(size, int)
            N = range(max(size))
            for dim in itertools.product(N, N):
                try:
                    out[dim] = self.generator.randint(low, high - 1)
                except IndexError:
                    continue
            return out
        else:
            raise TypeError(
                f"Expected a sequence of integers or a single integer, "
                f"got {size!r}"
            )

    def normal(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        size: None | int | _ShapeLike = None,
    ) -> ndarray | float:
        """Return random numbers from a normal (Gaussian) distribution.

        :param loc: Mean of the distribution, defaults to `0.0`.
        :param scale: Standard deviation of the distribution, defaults
            to `1.0`.
        :param size: Size of the output, defaults to `None`.
        :return: A new array populated with random numbers from a normal
            distribution.
        """
        if size is None:
            return self.generator.gauss(loc, scale)
        if isinstance(size, int):
            out = ndarray((size,))
            out[:] = [self.generator.gauss(loc, scale) for _ in range(size)]
            return out
        elif isinstance(size, tuple):
            out = ndarray(size)
            N = range(max(size))
            for dim in itertools.product(N, N):
                try:
                    out[dim] = self.generator.gauss(loc, scale)
                except IndexError:
                    continue
            return out
        else:
            raise TypeError(
                f"Expected a sequence of integers or a single integer, "
                f"got {size!r}"
            )


def default_rng(seed: None | int = None) -> Generator:
    """Construct a new Generator with the default BitGenerator.

    :param seed: Optional seed, defaults to `None`.
    :return: Generator instance with seed.
    """
    return Generator(BitGenerator(seed))
