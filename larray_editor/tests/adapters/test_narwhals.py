from larray_editor.arrayadapter import get_adapter
from larray_editor.tests.adapters.common import (adapter_values, needs_narwhals, needs_polars,
                                                 needs_pyarrow)


@needs_narwhals
@needs_polars
def test_narwhals_series_filter():
    import narwhals
    import polars

    series = narwhals.from_native(polars.Series('a', [3, 1, 3]), series_only=True)
    adapter = get_adapter(series)
    assert adapter.get_filter_options(0).tolist() == [1, 3]
    adapter.update_filter(0, [1])
    assert adapter_values(adapter) == [['', 'a'],
                                       [0, 3],
                                       [1, 3]]


@needs_narwhals
@needs_polars
def test_narwhals_dataframe():
    import narwhals
    import polars

    df = narwhals.from_native(polars.DataFrame({'a': [3, 1, 3], 'b': ['x', 'y', 'z']}))
    adapter = get_adapter(df)
    assert adapter_values(adapter) == [['', 'a', 'b'],
                                       [0, 3, 'x'],
                                       [1, 1, 'y'],
                                       [2, 3, 'z']]
    assert adapter.get_filter_options(0).tolist() == [1, 3]
    adapter.update_filter(0, [1])
    assert adapter_values(adapter) == [['', 'a', 'b'],
                                       [0, 3, 'x'],
                                       [1, 3, 'z']]


@needs_narwhals
@needs_polars
# the adapter converts to Pandas, which needs pyarrow for Polars-backed objects
@needs_pyarrow
def test_narwhals_missing_integers():
    import narwhals
    import polars

    # integers with missing values must not be displayed as floats
    df = polars.DataFrame({'i': [1, None], 's': ['a', 'b']})
    for obj in [narwhals.from_native(df), narwhals.from_native(df.lazy())]:
        assert adapter_values(obj) == [['', 'i', 's'],
                                       [0, 1, 'a'],
                                       [1, None, 'b']]
