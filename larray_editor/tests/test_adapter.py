import pytest

import larray as la


@pytest.fixture(scope="module")
def data():
    return la.ndtest((5, 5, 5, 5))


if __name__ == "__main__":
    pytest.main()
