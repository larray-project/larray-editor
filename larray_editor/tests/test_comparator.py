import os
from unittest import mock

import larray as la
import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from larray_editor.comparator import ArrayComparatorWindow, SessionComparatorWindow


# TODO: temporary, remove this once these tests are known to work on CI, which runs without
#       display (they should, using the offscreen Qt platform, see conftest.py)
pytestmark = pytest.mark.skipif(bool(os.environ.get('CI')),
                                reason="Qt tests are temporarily skipped on CI")

# the Qt application must stay referenced, otherwise it is garbage collected
qt_app = None


def compare_and_errors(window_class, data):
    """Open a comparator window on data and return it with the errors it displayed"""
    global qt_app
    qt_app = QApplication.instance() or QApplication([])
    with mock.patch('larray_editor.comparator.QMessageBox') as message_box, \
            mock.patch('larray_editor.comparator.print_exception'):
        window = window_class(data)
    errors = [call.args[2] for call in message_box.critical.call_args_list]
    return window, errors


def test_compare_arrays_with_anonymous_axis():
    arr = la.ndtest((30, la.Axis('sex=M,F')))
    window, errors = compare_and_errors(ArrayComparatorWindow, [arr, arr * 2])
    assert errors == []
    compared_shape = window.comparator_widget._combined_array.shape
    assert compared_shape == (30, 2, 2)


def test_compare_sessions_with_anonymous_axis():
    arr = la.ndtest((30, la.Axis('sex=M,F')))
    window, errors = compare_and_errors(SessionComparatorWindow,
                                        [la.Session(arr=arr), la.Session(arr=arr.copy())])
    assert errors == []
    # not displayed as different
    item_color = window.listwidget.item(0).foreground().color()
    assert item_color == Qt.black


def test_compare_arrays_with_anonymous_axes_of_the_same_length():
    # comparing those after stacking them is not supported yet, so they keep the align error
    # instead of failing later
    arr = la.Array(np.zeros((2, 2)))
    window, errors = compare_and_errors(ArrayComparatorWindow, [arr, arr * 2])
    assert errors == ['arrays with anonymous axes are currently not supported by Array.align']
