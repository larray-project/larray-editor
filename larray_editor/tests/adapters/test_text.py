from larray_editor.arrayadapter import get_adapter
from larray_editor.tests.adapters.common import adapter_values


def test_empty_csv_file(tmp_path):
    fpath = tmp_path / 'empty.csv'
    fpath.write_text('')
    assert adapter_values(fpath) == [['']]


def test_text_file_larger_than_first_indexed_chunk(tmp_path):
    # the adapter first indexes only 64 KB of the file, so indexing the rest must resume after
    # the last indexed line end instead of indexing it again (adding a phantom line)
    fpath = tmp_path / 'larger_than_64KB.txt'
    fpath.write_text(''.join(f'line{i:05d}\n' for i in range(10_000)))
    adapter = get_adapter(fpath)
    adapter.get_vlabels_values(0, 10_000)
    num_lines = adapter.shape2d()[0]
    adapter.close()
    assert num_lines == 10_000
