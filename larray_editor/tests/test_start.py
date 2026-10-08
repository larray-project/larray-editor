from larray_editor.start import join_split_path


def test_join_split_path(tmp_path):
    path_with_spaces = tmp_path / 'dir with spaces' / 'my file.h5'
    path_with_spaces.parent.mkdir()
    path_with_spaces.touch()
    split_path = str(path_with_spaces).split(' ')
    assert join_split_path(split_path) == [str(path_with_spaces)]

    # several existing paths are kept separate
    path1 = tmp_path / 'a.h5'
    path2 = tmp_path / 'b.h5'
    path1.touch()
    path2.touch()
    assert join_split_path([str(path1), str(path2)]) == [str(path1), str(path2)]
