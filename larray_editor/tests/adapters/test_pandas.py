import pandas as pd

from larray_editor.arrayadapter import get_adapter
from larray_editor.tests.adapters.common import adapter_values


def test_pandas_filter():
    adapter = get_adapter(pd.DataFrame({'a': [1, 2, 1]}))
    assert adapter.get_filter_options(0).tolist() == [1, 2]
    adapter.update_filter(0, [0])
    assert adapter_values(adapter) == [[None, 'a'],
                                       [0, 1],
                                       [2, 1]]
