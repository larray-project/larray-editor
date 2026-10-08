from larray_editor.tests.adapters.common import adapter_values


def test_empty_csv_file(tmp_path):
    fpath = tmp_path / 'empty.csv'
    fpath.write_text('')
    assert adapter_values(fpath) == [['']]
