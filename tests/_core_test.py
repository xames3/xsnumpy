import numpy as np
import pytest

import xsnumpy as xp

array_0r1c = xp.array([*range(10)])
array_1rnc = xp.array([[*range(10)]])
array_2r1c = xp.array([[1], [1]])
array_2rnc = xp.array([[*range(10)], [*range(10)]])
array_3t1r1c = xp.array([[[1]], [[1]], [[1]]])
array_3t1rnc = xp.array([[[*range(10)]], [[*range(10)]], [[*range(10)]]])
array_3t2r1c = xp.array([[[1], [1]], [[1], [1]], [[1], [1]]])
array_3t2rnc = xp.array(
    [
        [[*range(10)], [*range(10)]],
        [[*range(10)], [*range(10)]],
        [[*range(10)], [*range(10)]],
    ]
)


@pytest.mark.parametrize(
    ("array", "shape"),
    (
        (array_0r1c, (10,)),
        (array_1rnc, (1, 10)),
        (array_2r1c, (2, 1)),
        (array_2rnc, (2, 10)),
        (array_3t1r1c, (3, 1, 1)),
        (array_3t1rnc, (3, 1, 10)),
        (array_3t2r1c, (3, 2, 1)),
        (array_3t2rnc, (3, 2, 10)),
    ),
)
def test_array_shapes(array, shape):
    assert array.shape == shape


@pytest.mark.parametrize(
    ("array", "strides"),
    (
        (array_0r1c, (4,)),
        (array_1rnc, (40, 4)),
        (array_2r1c, (4, 4)),
        (array_2rnc, (40, 4)),
        (array_3t1r1c, (4, 4, 4)),
        (array_3t1rnc, (40, 40, 4)),
        (array_3t2r1c, (8, 4, 4)),
        (array_3t2rnc, (80, 40, 4)),
    ),
)
def test_array_strides(array, strides):
    assert array.strides == strides


@pytest.mark.parametrize(
    ("array", "ndim"),
    (
        (array_0r1c, 1),
        (array_1rnc, 2),
        (array_2r1c, 2),
        (array_2rnc, 2),
        (array_3t1r1c, 3),
        (array_3t1rnc, 3),
        (array_3t2r1c, 3),
        (array_3t2rnc, 3),
    ),
)
def test_array_ndim(array, ndim):
    assert array.ndim == ndim


@pytest.mark.parametrize(
    ("array", "size"),
    (
        (array_0r1c, 10),
        (array_1rnc, 10),
        (array_2r1c, 2),
        (array_2rnc, 20),
        (array_3t1r1c, 3),
        (array_3t1rnc, 30),
        (array_3t2r1c, 6),
        (array_3t2rnc, 60),
    ),
)
def test_array_size(array, size):
    assert array.size == size


@pytest.mark.parametrize(
    ("array", "itemsize"),
    (
        (array_0r1c, 4),
        (array_1rnc, 4),
        (array_2r1c, 4),
        (array_2rnc, 4),
        (array_3t1r1c, 4),
        (array_3t1rnc, 4),
        (array_3t2r1c, 4),
        (array_3t2rnc, 4),
    ),
)
def test_array_itemsize(array, itemsize):
    assert array.itemsize == itemsize


@pytest.mark.parametrize(
    ("array", "nbytes"),
    (
        (array_0r1c, 40),
        (array_1rnc, 40),
        (array_2r1c, 8),
        (array_2rnc, 80),
        (array_3t1r1c, 12),
        (array_3t1rnc, 120),
        (array_3t2r1c, 24),
        (array_3t2rnc, 240),
    ),
)
def test_array_nbytes(array, nbytes):
    assert array.nbytes == nbytes
