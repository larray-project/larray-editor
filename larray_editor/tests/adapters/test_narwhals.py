from larray_editor.arrayadapter import get_adapter
from larray_editor.tests.adapters.common import adapter_values, needs_narwhals, needs_polars


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
