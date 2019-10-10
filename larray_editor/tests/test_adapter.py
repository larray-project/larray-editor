import pytest

import numpy as np
import larray as la

from larray_editor.utils import Product
from larray_editor.arraymodel import DataArrayModel, LARGE_NROWS, LARGE_COLS


@pytest.fixture(scope="module")
def data():
    return la.ndtest((5, 5, 5, 5))


if __name__ == "__main__":
    pytest.main()
