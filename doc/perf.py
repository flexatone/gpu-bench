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

#-------------------------------------------------------------------------------

class APScale(ArrayProcessor):
    NAME = 'scaling'
    SORT = 0

    def __call__(self):
        self.array * 100.0


class APSum(ArrayProcessor):
    NAME = 'scaling'
    SORT = 0

    def __call__(self):
        self.array * 100.0


#-------------------------------------------------------------------------------

class FixtureFactory(Fixture):
    NAME = ''

    @classmethod
    def get_label_array(cls, size: int) -> tp.Tuple[str, np.ndarray]:
        array = cls.get_array(size)
        return cls.NAME, array



class FFNumPyInt64(FixtureFactory):
    NAME = 'np-int64'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return np.arange(size)


class FFCuPyInt64(FixtureFactory):
    NAME = 'cp-int64'

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return cp.arange(size)




CLS_PROCESSOR = (
    APScale,
    APSum,
    )

CLS_FF = (
    FFNumPyInt64,
    FFCuPyInt64,
)


SIZES = (100_000, 1_000_000, 10_000_000)

if __name__ == '__main__':

    directory = Path('doc/')

    for fn, title, processors in (
        ('fig-0.png', 'Performance', CLS_PROCESSOR),
    ):
        run_test(sizes=SIZES,
                fixtures=CLS_FF,
                processors=processors,
                fp=directory / fn,
                title=title,
                number=100,
                )


