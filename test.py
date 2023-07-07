import cupy as cp
import numpy as np


def test_dtypes():
    for dtype in (np.int64, np.int32, np.int16, np.int8):
        a = cp.arange(8, dtype=dtype)
        print(a)