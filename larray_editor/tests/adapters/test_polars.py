import datetime as dt
import warnings

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


@needs_polars
def test_polars_filter_datetime_column():
    import polars

    jan1 = dt.datetime(2024, 1, 1)
    df = polars.DataFrame({'d': [jan1, dt.datetime(2024, 1, 2), jan1]})
    for obj in [df, df.lazy()]:
        adapter = get_adapter(obj)
        assert len(adapter.get_filter_options(0)) == 2
        # polars emits some warnings from Rust code which cannot be turned
        # into errors, so we need to record them
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter('always')
            adapter.update_filter(0, [0])
        assert not recorded_warnings
        assert adapter_values(adapter) == [['', 'd'],
                                           [0, jan1],
                                           [1, jan1]]


@needs_polars
def test_polars_filter_null():
    import polars

    series = polars.Series('a', [1, None, 1])
    for obj in [series, series.to_frame(), series.to_frame().lazy()]:
        adapter = get_adapter(obj)
        # nulls are sorted first
        assert len(adapter.get_filter_options(0)) == 2
        adapter.update_filter(0, [0])
        assert adapter.shape2d() == (1, 1)
