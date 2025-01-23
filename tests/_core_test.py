import numpy as np
import pytest

import xsnumpy as xp


def make_arrays(params):
    values = []
    for module in (xp, np):
        values.append(getattr(module, "array")(params))
    return values


xarray_0r1c, narray_0r1c = make_arrays(params=[*range(10)])
xarray_1rnc, narray_1rnc = make_arrays(params=[[*range(10)]])
xarray_2r1c, narray_2r1c = make_arrays(params=[[1] * 2])
xarray_2rnc, narray_2rnc = make_arrays(params=[[*range(10)] * 2])
xarray_3t1r1c, narray_3t1r1c = make_arrays(params=[[[1]], [[1]], [[1]]])
xarray_3t1rnc, narray_3t1rnc = make_arrays(params=[[[*range(10)]] * 3])
xarray_3t2r1c, narray_3t2r1c = make_arrays(params=[[[1], [1]] * 3])
xarray_3t2rnc, narray_3t2rnc = make_arrays(
    params=[[[*range(10)], [*range(10)]] * 3]
)


@pytest.mark.parametrize(
    ("xarray", "narray"),
    (
        (xarray_0r1c, narray_0r1c),
        (xarray_1rnc, narray_1rnc),
        (xarray_2r1c, narray_2r1c),
        (xarray_2rnc, narray_2rnc),
        (xarray_3t1r1c, narray_3t1r1c),
        (xarray_3t1rnc, narray_3t1rnc),
        (xarray_3t2r1c, narray_3t2r1c),
        (xarray_3t2rnc, narray_3t2rnc),
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
