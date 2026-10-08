import pandas as pd

from larray_editor.tests.adapters.common import adapter_values


def test_no_column():
    df_without_column = pd.DataFrame(index=[1, 2])
    assert adapter_values(df_without_column) == [[None],
                                                 [1],
                                                 [2]]
