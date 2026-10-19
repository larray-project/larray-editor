import datetime as dt
import logging
import os

import larray as la
import pandas as pd
import pytest
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from larray_editor import arraywidget
from larray_editor.arraywidget import ArrayEditorWidget
from larray_editor.utils import logger


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


def test_empty_array_logs_no_error():
    logged_errors = []
    handler = logging.Handler(level=logging.ERROR)
    handler.emit = logged_errors.append
    logger.addHandler(handler)
    try:
        create_array_widget(la.Array([]))
    finally:
        logger.removeHandler(handler)
    assert logged_errors == []


def test_complex_values():
    model = create_array_widget(la.Array([1.5 + 2j, 0.5 - 1j], 'x=x0,x1')).model_data
    displayed_texts = [model.data(model.index(0, col), Qt.DisplayRole) for col in range(2)]
    assert displayed_texts == ['1.5+2.0j', '0.5-1.0j']


def test_row_labels_column_fits_labels_without_column():
    df_without_column = pd.DataFrame(index=['a_long_row_label', 'another_long_row_label'])
    view = create_array_widget(df_without_column).view_vlabels
    labels_width = view.sizeHintForColumn(0)
    assert view.columnWidth(0) >= labels_width


def test_row_labels_view_width_when_auto_width_is_default_width():
    arr_2d = la.ndtest((2, 2))
    auto_width = create_array_widget(arr_2d).view_vlabels.columnWidth(0)
    # the bug only happened when the auto width of the new array was the default width
    orig_default_width = arraywidget.DEFAULT_COLUMN_WIDTH
    arraywidget.DEFAULT_COLUMN_WIDTH = auto_width
    try:
        widget = create_array_widget(la.ndtest('a=a0'))
        widget.set_data(arr_2d)
    finally:
        arraywidget.DEFAULT_COLUMN_WIDTH = orig_default_width
    view = widget.view_vlabels
    columns_width = view.verticalHeader().width() + view.columnWidth(0)
    assert view.width() == columns_width


def test_shift_click_extends_selection():
    view = create_array_widget(la.ndtest((5, 5))).view_data

    def click(row, column, modifier=Qt.NoModifier):
        cell_center = view.visualRect(view.model().index(row, column)).center()
        QTest.mouseClick(view.viewport(), Qt.LeftButton, modifier, cell_center)

    click(1, 1)
    click(3, 3, Qt.ShiftModifier)
    # row_min, row_max, col_min, col_max (exclusive max)
    assert view.selection_bounds() == (1, 4, 1, 4)
