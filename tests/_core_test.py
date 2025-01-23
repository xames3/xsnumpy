import numpy as np
import pytest

import xsnumpy as xp

np.set_printoptions(4)
xp.set_printoptions(4)


def make_arrays(params):
    values = []
    for module in (xp, np):
        values.append(getattr(module, "array")(params))
    return values


@pytest.mark.parametrize(
    ("xarray", "narray"),
    (
        make_arrays([*range(10)]),
        make_arrays([[*range(10)]]),
        make_arrays([[1] * 2]),
        make_arrays([[*range(10)] * 2]),
        make_arrays([[[1]], [[1]], [[1]]]),
        make_arrays([[[*range(10)]] * 3]),
        make_arrays([[[1], [1]] * 3]),
        make_arrays([[[*range(10)], [*range(10)]] * 3]),
    ),
)
def test_attributes(xarray, narray):
    assert xarray.shape == narray.shape
    assert xarray.strides == narray.strides
    assert xarray.ndim == narray.ndim
    assert xarray.ndim == narray.ndim
    assert xarray.size == narray.size
    assert xarray.itemsize == narray.itemsize
    assert xarray.nbytes == narray.nbytes
    assert xarray.__str__().replace("\n", "").replace(
        " ", ""
    ) == narray.__str__().replace("\n", "").replace(" ", "")


def test_arange():
    assert (xp.arange(2) == np.arange(2)).all()
    assert (xp.arange(5.0) == np.arange(5.0)).all()
    assert (xp.arange(1, 6) == np.arange(1, 6)).all()
    assert (xp.arange(1, 7.5) == np.arange(1, 7.5)).all()
    assert (xp.arange(3, 9, 3) == np.arange(3, 9, 3)).all()
    assert (xp.arange(1, 10, 1.5) == np.arange(1, 10, 1.5)).all()
    assert (xp.arange(2.5, 7.5, 3) == np.arange(2.5, 7.5, 3)).all()


def test_zeros():
    assert xp.zeros(2).size == np.zeros(2).size
    assert xp.zeros(5).size == np.zeros(5).size
    assert xp.zeros(1, 6).size == np.zeros((1, 6)).size
    assert xp.zeros(1, 7).size == np.zeros((1, 7)).size
    assert xp.zeros(3, 9, 3).size == np.zeros((3, 9, 3)).size
    assert xp.zeros(1, 10, 1).size == np.zeros((1, 10, 1)).size
    assert xp.zeros(2, 7, 3).size == np.zeros((2, 7, 3)).size


def test_ones():
    assert (xp.ones(2) == np.ones(2)).all()
    assert (xp.ones(5) == np.ones(5)).all()
    assert (xp.ones(1, 6) == np.ones((1, 6))).all()
    assert (xp.ones(1, 7) == np.ones((1, 7))).all()
    assert (xp.ones(3, 9, 3) == np.ones((3, 9, 3))).all()
    assert (xp.ones(1, 10, 1) == np.ones((1, 10, 1))).all()
    assert (xp.ones(2, 7, 3) == np.ones((2, 7, 3))).all()


def test_full():
    assert (xp.full(2, fill_value=2.0) == np.full(2, 2.0)).all()
    assert (xp.full(1, 3, fill_value=3.14) == np.full((1, 3), 3.14)).all()
    assert (xp.full(3, 2, 3, fill_value=7.5) == np.full((3, 2, 3), 7.5)).all()


@pytest.mark.parametrize(
    ("arrays"),
    (
        (make_arrays([*range(10)])),
        make_arrays((1, 3.5, 4.2, 2.3, 2.1, 4.3)),
        make_arrays((4.95, 0.54, 2.45, 3.12, 4.23, 0.12, 1.56, 2.75)),
    ),
)
def test_arithmetic_add_sub(arrays):
    assert ((arrays[0] + arrays[0]) == (arrays[1] + arrays[1])).all()
    assert ((arrays[0] - arrays[0]) == (arrays[1] - arrays[1])).all()
    assert ((arrays[0] + xp.ones(1)) == (arrays[1] + np.ones(1))).all()


@pytest.mark.parametrize(
    ("arrays"),
    (
        (make_arrays([*range(1, 6)])),
        make_arrays((1, 4, 2, 5, 2, 2, 7, 3, 3)),
    ),
)
def test_arithmetic_mul_div(arrays):
    assert ((arrays[0] * arrays[0]) == (arrays[1] * arrays[1])).all()
    assert ((arrays[0] / arrays[0]) == (arrays[1] / arrays[1])).all()


@pytest.mark.parametrize(
    ("arrays"),
    (
        (make_arrays([*range(6)])),
        make_arrays((1, 4, 2, -5, 3.2)),
        make_arrays((-0.2, 3.2, 1.7, 4, 3.2, -0.43, -2.4)),
    ),
)
def test_logical_comparison(arrays):
    assert ((arrays[0] < 3) == (arrays[1] < 3)).all()
    assert ((arrays[0] <= 2) == (arrays[1] <= 2)).all()
    assert ((arrays[0] >= 1) == (arrays[1] >= 1)).all()
    assert ((arrays[0] > 2) == (arrays[1] > 2)).all()
