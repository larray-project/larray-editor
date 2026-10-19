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


def test_text_file_crlf_line_end_straddling_chunks(tmp_path):
    # with lines of 1985 bytes, a \r\n straddles the boundary of the 4 MB chunks indexed after
    # the first 64 KB, because 4 MB + 1 is a multiple of 1985
    num_lines = 2200
    fpath = tmp_path / 'crlf.txt'
    fpath.write_bytes((b'x' * 1983 + b'\r\n') * num_lines)
    adapter = get_adapter(fpath)
    adapter.get_vlabels_values(0, num_lines)
    num_indexed_lines = adapter.shape2d()[0]
    adapter.close()
    assert num_indexed_lines == num_lines
