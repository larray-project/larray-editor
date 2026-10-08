import os

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
