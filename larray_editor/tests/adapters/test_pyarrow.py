from larray_editor.tests.adapters.common import adapter_values, needs_pyarrow


@needs_pyarrow
def test_feather_file_without_row(tmp_path):
    import pyarrow
    import pyarrow.feather

    fpath = tmp_path / 'empty.feather'
    pyarrow.feather.write_feather(pyarrow.table({'a': pyarrow.array([], pyarrow.int64())}), fpath)
    assert adapter_values(fpath) == [['', 'a']]
