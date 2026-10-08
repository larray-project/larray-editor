import datetime as dt

import numpy as np

from larray_editor.tests.adapters.common import adapter_values


def test_timedelta_values():
    durations = np.array([1, 2], dtype='timedelta64[h]')
    assert adapter_values(durations) == [[0, 1],
                                         [dt.timedelta(hours=1), dt.timedelta(hours=2)]]
