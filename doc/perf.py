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
    NAME = 'scale-np'
    SORT = 0

    def __call__(self):
        self.array * 100.0


class APScaleCP(ArrayProcessorCP):
    NAME = 'scale-cp'
    SORT = 0

    def __call__(self):
        self.array * 100.0


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

# class FFCuPyInt64(FixtureFactory):
#     NAME = 'cp-int64'

#     @staticmethod
#     def get_array(size: int) -> np.ndarray:
#         return cp.arange(size)




CLS_PROCESSOR = (
    APScaleNP,
    APScaleCP,
    APSumNP,
    APSumCP,
    APSelectNP,
    APSelectCP,
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
        ('fig-0.png', 'Performance (Log Scale)', CLS_PROCESSOR),
    ):
        run_test(sizes=SIZES,
                fixtures=CLS_FF,
                processors=processors,
                fp=directory / fn,
                title=title,
                number=10,
                )


