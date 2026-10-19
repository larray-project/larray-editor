import datetime as dt
import warnings
from unittest import mock

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
        assert adapter_values(adapter) == [['', 'a'],
                                           [0, None]]


@needs_polars
def test_polars_missing_integers():
    import polars

    # integers with missing values must not be displayed as floats
    df = polars.DataFrame({'i': [1, None], 's': ['a', 'b']})
    for obj in [df, df.lazy()]:
        assert adapter_values(obj) == [['', 'i', 's'],
                                       [0, 1, 'a'],
                                       [1, None, 'b']]
    series = polars.Series('i', [1, None])
    assert adapter_values(series) == [['', 'i'],
                                      [0, 1],
                                      [1, None]]


@needs_polars
def test_polars_filter_select_all_truncated_options():
    import polars

    df = polars.DataFrame({'a': [1, 2, 3, 4, 5]})
    for obj in [df, df.lazy(), df['a']]:
        adapter = get_adapter(obj)
        with mock.patch('larray_editor.arrayadapter.MAX_FILTER_OPTIONS', 3):
            assert len(adapter.get_filter_options(0)) == 3
            adapter.update_filter(0, [0])
            # selecting all the (truncated) options
            adapter.update_filter(0, [0, 1, 2])
            assert adapter.shape2d()[0] == 5
            # selecting none of them
            adapter.update_filter(0, [])
            assert adapter.shape2d()[0] == 5
