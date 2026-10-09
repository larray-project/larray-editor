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


def test_pandas_missing_integers():
    # integers with missing values must not be displayed as floats
    df = pd.DataFrame({'i': pd.array([1, None], dtype='Int64'), 's': ['a', 'b']})
    assert adapter_values(df) == [[None, 'i', 's'],
                                  [0, 1, 'a'],
                                  [1, None, 'b']]
    assert adapter_values(df['i']) == [[None, ''],
                                       [0, 1],
                                       [1, None]]
