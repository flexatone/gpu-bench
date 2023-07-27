import os
import sys
import typing as tp
from pathlib import Path
import numpy as np
import cupy as cp

sys.path.append(os.getcwd())

from plot import run_test
from plot import ArrayProcessor
from plot import Fixture


class ArrayProcessorCP(ArrayProcessor):
    NAME = ''
    SORT = -1

    def __init__(self, array: np.ndarray):
        self.array = cp.array(array)


#-------------------------------------------------------------------------------

class APScaleNP(ArrayProcessor):
    NAME = 'scale1x-np'
    SORT = 0

    def __call__(self):
        self.array * 100.0

class APScaleCP(ArrayProcessorCP):
    NAME = 'scale1x-cp'
    SORT = 0

    def __call__(self):
        self.array * 100.0

class APScaleNPCP(ArrayProcessor):
    NAME = 'scale1x-np-cp'
    SORT = 0

    def __call__(self):
        cp.array(self.array) * 100.0



class APScale4xNP(ArrayProcessor):
    NAME = 'scale4x-np'
    SORT = 0

    def __call__(self):
        self.array * 100.0 * 0.5 * 4 * 0.1

class APScale4xCP(ArrayProcessorCP):
    NAME = 'scale4x-cp'
    SORT = 0

    def __call__(self):
        self.array * 100.0 * 0.5 * 4 * 0.1

class APScale4xNPCP(ArrayProcessor):
    NAME = 'scale4x-np-cp'
    SORT = 0

    def __call__(self):
        cp.array(self.array) * 100.0 * 0.5 * 4 * 0.1




class APSumNP(ArrayProcessor):
    NAME = 'sum-np'
    SORT = 1

    def __call__(self):
        self.array.sum()

class APSumCP(ArrayProcessorCP):
    NAME = 'sum-cp'
    SORT = 1

    def __call__(self):
        self.array.sum()

class APSumNPCP(ArrayProcessor):
    NAME = 'sum-np-cp'
    SORT = 1

    def __call__(self):
        cp.array(self.array).sum()



class APSelectNP(ArrayProcessor):
    NAME = 'select-bool-np'
    SORT = 1

    def __call__(self):
        self.array[self.array % 2 == 0]

class APSelectCP(ArrayProcessorCP):
    NAME = 'select-bool-cp'
    SORT = 1

    def __call__(self):
        self.array[self.array % 2 == 0]

class APSelectNPCP(ArrayProcessor):
    NAME = 'select-bool-np-cp'
    SORT = 1

    def __call__(self):
        a = cp.array(self.array)
        a[a % 2 == 0]



class APSliceNP(ArrayProcessor):
    NAME = 'select-slice-np'
    SORT = 1

    def __call__(self):
        self.array[:len(self.array) // 2]

class APSliceCP(ArrayProcessorCP):
    NAME = 'select-slice-cp'
    SORT = 1

    def __call__(self):
        self.array[:len(self.array) // 2]

class APSliceNPCP(ArrayProcessor):
    NAME = 'select-slice-np-cp'
    SORT = 1

    def __call__(self):
        a = cp.array(self.array)
        a[:len(a) // 2]



class APArgsortNP(ArrayProcessor):
    NAME = 'argsort-np'
    SORT = 1

    def __call__(self):
        np.argsort(self.array)

class APArgsortCP(ArrayProcessorCP):
    NAME = 'argsort-cp'
    SORT = 1

    def __call__(self):
        cp.argsort(self.array)

class APArgsortNPCP(ArrayProcessor):
    NAME = 'argsort-np-cp'
    SORT = 1

    def __call__(self):
        cp.argsort(cp.array(self.array))


#-------------------------------------------------------------------------------

class FixtureFactory(Fixture):
    NAME = ''

    @classmethod
    def get_label_array(cls, size: int) -> tp.Tuple[str, np.ndarray]:
        array = cls.get_array(size)
        return cls.NAME, array



class FFInt64(FixtureFactory):
    NAME = 'int64'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return np.arange(size, dtype=np.int64)

class FFFloat64(FixtureFactory):
    NAME = 'float64'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return np.arange(size, dtype=np.float64) * 0.5

class FFBool(FixtureFactory):
    NAME = 'bool'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return (np.arange(size) % 3).astype(bool)


CLS_MATH = (
    APScaleNP,
    APScaleCP,
    APScaleNPCP,
    APScale4xNP,
    APScale4xCP,
    APScale4xNPCP,
    APSumNP,
    APSumCP,
    APSumNPCP,
    )


CLS_SELECT = (
    APSelectNP,
    APSelectCP,
    APSelectNPCP,
    APSliceNP,
    APSliceCP,
    APSliceNPCP,
    APArgsortNP,
    APArgsortCP,
    APArgsortNPCP,
    )


CLS_FF = (
    FFInt64,
    FFFloat64,
    FFBool,
)


SIZES = (100_000, 1_000_000, 10_000_000)

if __name__ == '__main__':

    directory = Path('doc/')

    for fn, title, processors in (
        ('fig-0.png', 'Math (Log Scale)', CLS_MATH),
        ('fig-1.png', 'Math (Linear Scale)', CLS_MATH),
        ('fig-2.png', 'Selection (Log Scale)', CLS_SELECT),
        ('fig-3.png', 'Selection (Linear Scale)', CLS_SELECT),

    ):
        run_test(sizes=SIZES,
                fixtures=CLS_FF,
                processors=processors,
                fp=directory / fn,
                title=title,
                number=100,
                )


