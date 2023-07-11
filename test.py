import cupy as cp
import numpy as np


def test_dtypes():
    for dtype in (np.int64, np.int32, np.int16, np.int8):
        a = cp.arange(8, dtype=dtype)
        # they give back numpy dtypes
        assert a.dtype == dtype

    for dtype in (np.uint64, np.uint32, np.uint16, np.uint8):
        a = cp.arange(8, dtype=dtype)
        assert a.dtype == dtype

    for dtype in (np.float64, np.float32, np.float16):
        a = cp.arange(8, dtype=dtype)
        assert a.dtype == dtype

    for dtype in (np.complex64, np.complex128):
        a = cp.arange(8, dtype=dtype)
        assert a.dtype == dtype

    a = cp.array([False, True, False], dtype=bool)
    assert  a.tolist() == [False, True, False]

    # no support for object, unicode, dt64

def test_concat():
    assert (cp.concatenate((cp.array([False, True]), cp.array([10, 20])), axis=0).tolist() ==
            [0, 1, 10, 20])

    assert (cp.concatenate((cp.array([3.1, 5.1]), cp.array([10, 20])), axis=0).tolist() ==
            [3.1, 5.1, 10, 20])

def test_frombuffer():
    assert cp.frombuffer(cp.array([3,4,5], dtype=np.int64).tobytes(), dtype=np.int64).tolist() == [3, 4, 5]


def test_binop():
    # cannot vectorwise op with NumPy arrays
    # cp.array((3, -3)) * np.array((1, 10))
    assert (cp.array((3, -3)) * cp.array((1, 10))).tolist() == [3, -30]


def test_class():
    # different classes, not instances of each other
    assert np.array([3, 2]).__class__ != cp.array([3, 2]).__class__


    # import ipdb; ipdb.set_trace()

# hit a memory limit...
# ipdb> np.empty((100000, 100000))
# *** numpy.core._exceptions._ArrayMemoryError: Unable to allocate 74.5 GiB for an array with shape (100000, 100000) and data type float64

# >>> cp.arange(3).flags
#   C_CONTIGUOUS : True
#   F_CONTIGUOUS : True
#   OWNDATA : True

