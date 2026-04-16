import pandas as pd
import pytest
from larray.tests.common import needs_pytables

from larray_editor.tests.adapters.common import hdf_adapter_values


@needs_pytables
def test_pytables_pandas_frame_mixed_type_index(tmp_path):
    index = pd.Index(['x', 1], dtype=object, name='i')
    with pytest.warns(pd.errors.PerformanceWarning, match='pickle object types'):
        rows = hdf_adapter_values(pd.DataFrame({'c': [1, 2]}, index=index), tmp_path)
    assert rows == [['i', 'c'],
                    ['x', 1],
                    [1, 2]]
