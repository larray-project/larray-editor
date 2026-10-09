from larray_editor.tests.adapters.common import adapter_values, needs_pyarrow


@needs_pyarrow
def test_feather_file_without_row(tmp_path):
    import pyarrow
    import pyarrow.feather

    fpath = tmp_path / 'empty.feather'
    pyarrow.feather.write_feather(pyarrow.table({'a': pyarrow.array([], pyarrow.int64())}), fpath)
    assert adapter_values(fpath) == [['', 'a']]


@needs_pyarrow
def test_pyarrow_missing_integers():
    import pyarrow

    # integers with missing values must not be displayed as floats
    table = pyarrow.table({'i': pyarrow.array([1, None]), 's': ['a', 'b']})
    assert adapter_values(table) == [['', 'i', 's'],
                                     [0, 1, 'a'],
                                     [1, None, 'b']]
    array = pyarrow.array([1, None])
    assert adapter_values(array) == [['', 'value'],
                                     [0, 1],
                                     [1, None]]
