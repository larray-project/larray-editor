import pytest

import numpy as np

from larray import ndtest, zeros, LArray
from larray_editor.utils import Product
from larray_editor.arraymodel import DataArrayModel, LARGE_NROWS, LARGE_COLS

@pytest.fixture(scope="module")
def data():
    return ndtest((5, 5, 5, 5))


if __name__ == "__main__":
    pytest.main()
