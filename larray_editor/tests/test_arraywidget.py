import datetime as dt
import os

import larray as la
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from larray_editor.arraywidget import ArrayEditorWidget


# TODO: temporary, remove this once these tests are known to work on CI, which runs without
#       display (they should, using the offscreen Qt platform, see conftest.py)
pytestmark = pytest.mark.skipif(bool(os.environ.get('CI')),
                                reason="Qt tests are temporarily skipped on CI")

# the Qt application must stay referenced, otherwise it is garbage collected
qt_app = None


def create_array_widget(data):
    """Create an array widget (and the Qt application if needed) displaying data"""
    global qt_app
    qt_app = QApplication.instance() or QApplication([])
    widget = ArrayEditorWidget(None)
    widget.set_data(data)
    return widget


def test_drag_selection_on_labels_going_back():
    view = create_array_widget(la.ndtest((6, 3))).view_data
    selection_model = view.selectionModel()

    # dragging the mouse on the row labels from row 1 to row 4, then back to row 3
    view.selectRow(1)
    for row in [2, 3, 4, 3]:
        view.selectNewRow(row)
    assert [row for row in range(6) if selection_model.isRowSelected(row)] == [1, 2, 3]

    # same on the column labels from column 0 to column 2, then back to column 1
    view.selectColumn(0)
    for column in [1, 2, 1]:
        view.selectNewColumn(column)
    assert [col for col in range(3) if selection_model.isColumnSelected(col)] == [0, 1]


def test_pandas_series_datetime():
    series = pd.Series([dt.datetime(2024, 1, 1)])
    model = create_array_widget(series).model_data
    displayed_text = model.data(model.index(0, 0), Qt.DisplayRole)
    # like a DataFrame column (and Pandas), not like a numpy datetime64
    assert displayed_text == '2024-01-01 00:00:00'
