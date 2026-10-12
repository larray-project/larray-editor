import os
from unittest import mock

import larray as la
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from larray_editor.editor import MappingEditorWindow, qtconsole_available


# TODO: temporary, remove this once these tests are known to work on CI, which runs without
#       display (they should, using the offscreen Qt platform, see conftest.py)
pytestmark = pytest.mark.skipif(bool(os.environ.get('CI')),
                                reason="Qt tests are temporarily skipped on CI")

needs_qtconsole = (
    pytest.mark.skipif(not qtconsole_available,
                       reason="qtconsole is required for this test")
)

# the Qt application must stay referenced, otherwise it is garbage collected
qt_app = None


def create_editor(data, **kwargs):
    """Create a session editor window (and the Qt application if needed)"""
    global qt_app
    qt_app = QApplication.instance() or QApplication([])
    return MappingEditorWindow(data, **kwargs)


def run_command(editor, command):
    editor.ipython_kernel.shell.run_cell(command)
    editor.ipython_cell_executed()


def displayed_value(editor, row, column):
    model = editor.array_widget.model_data
    return model.data(model.index(row, column), Qt.DisplayRole)


@needs_qtconsole
def test_refresh_current_object_modified_by_function():
    editor = create_editor(la.Session(arr=la.ndtest((2, 3))))
    editor.select_list_item('arr')
    assert displayed_value(editor, 1, 0) == '3'
    run_command(editor, "def update():\n    arr['a1'] = 0")
    run_command(editor, "update()")
    assert displayed_value(editor, 1, 0) == '0'


def test_readonly_editor():
    session = la.Session(arr=la.ndtest((2, 3)))
    editor = create_editor(session, readonly=True)
    editor.select_list_item('arr')
    model = editor.array_widget.model_data
    index = model.index(0, 0)
    assert not model.flags(index) & Qt.ItemIsEditable
    assert not model.setData(index, '42')
    assert editor.data['arr'].i[0, 0] == 0


def test_expression_readonly():
    editor = create_editor(la.Session(arr=la.ndtest((2, 3))))
    model = editor.array_widget.model_data
    editor.view_expr(editor.data['arr'] + 1, 'arr + 1')
    assert not model.flags(model.index(0, 0)) & Qt.ItemIsEditable
    editor.select_list_item('arr')
    assert model.flags(model.index(0, 0)) & Qt.ItemIsEditable


def test_edit_without_console():
    session = la.Session(arr=la.ndtest((2, 3)))
    editor = create_editor(session, python_console=False)
    editor.select_list_item('arr')
    model = editor.array_widget.model_data
    model.setData(model.index(0, 0), '100')
    assert editor.data['arr'].i[0, 0] == 100


def test_title_shows_unsaved_modifications():
    editor = create_editor(la.Session(arr=la.ndtest((2, 3))), python_console=False)
    editor.select_list_item('arr')
    model = editor.array_widget.model_data
    model.setData(model.index(0, 0), '100')
    assert editor.windowTitle().startswith('*')
    editor.edit_undo_stack.undo()
    assert not editor.windowTitle().startswith('*')


def test_paste_value_in_column_larger_than_buffer():
    editor = create_editor(la.Session(arr=la.ndtest((2000, 3))), python_console=False)
    editor.select_list_item('arr')
    array_widget = editor.array_widget
    array_widget.view_data.selectColumn(1)
    QApplication.clipboard().setText('42')
    array_widget.paste()
    pasted_column = editor.data['arr'].i[:, 1]
    assert (pasted_column == 42).all()


def paste(editor, corner1, corner2, text):
    """Paste text in the selection between two cells (global row, column positions)"""
    view = editor.array_widget.view_data
    view.first_selection_corner = corner1
    view.second_selection_corner = corner2
    QApplication.clipboard().setText(text)
    editor.array_widget.paste()


def test_paste_empty_clipboard():
    editor = create_editor(la.Session(arr=la.ndtest((2, 3))), python_console=False)
    editor.select_list_item('arr')
    paste(editor, (0, 0), (0, 0), '')
    assert editor.data['arr'].equals(la.ndtest((2, 3)))


def test_paste_lines_of_different_lengths():
    editor = create_editor(la.Session(arr=la.ndtest((2, 3))), python_console=False)
    editor.select_list_item('arr')
    with mock.patch('larray_editor.arraywidget.QMessageBox') as message_box:
        paste(editor, (0, 0), (0, 0), '1\t2\n3')
    message_box.critical.assert_called_once()
    assert editor.data['arr'].equals(la.ndtest((2, 3)))


def test_paste_past_the_last_row_and_column():
    editor = create_editor(la.Session(arr=la.ndtest((2, 3))), python_console=False)
    editor.select_list_item('arr')
    with mock.patch('larray_editor.arraywidget.QMessageBox') as message_box:
        paste(editor, (1, 2), (1, 2), '10\t11\n12\t13')
    message_box.warning.assert_called_once()
    # only the part which fits is pasted
    assert editor.data['arr'].data.tolist() == [[0, 1, 2],
                                                [3, 4, 10]]


def test_sorted_array_readonly():
    editor = create_editor(la.Session(arr=la.ndtest((3, 3))), python_console=False)
    editor.select_list_item('arr')
    model = editor.array_widget.model_data
    with mock.patch('larray_editor.arraymodel.QMessageBox'):
        # sorting rows by the values of a column
        editor.array_widget.sort_hlabel(0, 0, False)
        assert not model.flags(model.index(0, 0)) & Qt.ItemIsEditable
        assert not model.setData(model.index(0, 0), '100')
        paste(editor, (0, 0), (0, 0), '100')
        assert editor.data['arr'].equals(la.ndtest((3, 3)))

        # sorting the labels of an axis
        editor.select_list_item('arr')
        editor.array_widget.sort_axis_labels(0, False)
        assert not model.setData(model.index(0, 0), '100')
        assert editor.data['arr'].equals(la.ndtest((3, 3)))


def test_edit_string_value():
    editor = create_editor(la.Session(arr=la.Array(['a', 'b'], 'x=x0,x1')))
    editor.select_list_item('arr')
    model = editor.array_widget.model_data
    model.setData(model.index(0, 0), 'c')
    assert editor.data['arr'].i[0] == 'c'


def test_edit_object_value():
    editor = create_editor(la.Session(arr=la.Array(['a', 1], 'x=x0,x1', dtype=object)),
                           python_console=False)
    editor.select_list_item('arr')
    model = editor.array_widget.model_data
    # a number
    model.setData(model.index(0, 0), '100')
    edited_value = editor.data['arr'].i[0]
    assert edited_value == 100
    # not 100+0j (which compares equal to 100)
    assert isinstance(edited_value, int)
    # anything else is kept as a string
    model.setData(model.index(0, 1), 'hello')
    assert editor.data['arr'].i[1] == 'hello'


@needs_qtconsole
def test_undo_updates_color_scale():
    editor = create_editor(la.Session(arr=la.ndtest((2, 3))))
    editor.select_list_item('arr')
    model = editor.array_widget.model_data
    max_cell = model.index(1, 2)
    max_cell_color = model.data(max_cell, Qt.BackgroundRole)
    model.setData(model.index(0, 0), '100')
    assert model.data(max_cell, Qt.BackgroundRole) != max_cell_color
    editor.edit_undo_stack.undo()
    assert model.data(max_cell, Qt.BackgroundRole) == max_cell_color
