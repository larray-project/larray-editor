from larray_editor.arrayadapter import get_adapter
from larray_editor.tests.adapters.common import adapter_values, needs_polars


@needs_polars
def test_polars_lazyframe_filter():
    import polars

    # this used to emit a PerformanceWarning
    adapter = get_adapter(polars.DataFrame({'a': [1, 2, 1]}).lazy())
    assert adapter.get_filter_options(0).tolist() == [1, 2]
    adapter.update_filter(0, [0])
    assert adapter_values(adapter) == [['', 'a'],
                                       [0, 1],
                                       [1, 1]]
