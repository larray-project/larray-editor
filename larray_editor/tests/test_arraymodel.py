import os
import unittest

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from larray_editor.arrayadapter import AbstractAdapter
from larray_editor.arraymodel import DataArrayModel
from larray_editor.utils import logger


# TODO: temporary, remove this once these tests are known to work on CI, which runs without
#       display (they should, using the offscreen Qt platform, see conftest.py)
pytestmark = pytest.mark.skipif(bool(os.environ.get('CI')),
                                reason="Qt tests are temporarily skipped on CI")

# the Qt application must stay referenced, otherwise it is garbage collected
qt_app = None


class BuggyAdapter(AbstractAdapter):
    """Adapter returning raw_data as is, whatever its shapes"""
    def __init__(self, raw_data):
        super().__init__(data=None)
        self.raw_data = raw_data

    def shape2d(self):
        return 2, 2

    def get_data_values_and_attributes(self, h_start, v_start, h_stop, v_stop):
        return self.raw_data


def model_logged_errors(raw_data):
    """Display raw_data in a data model and return (model, errors logged while doing so)"""
    global qt_app
    qt_app = QApplication.instance() or QApplication([])
    model = DataArrayModel(adapter=None)
    with unittest.TestCase().assertLogs(logger, level='ERROR') as logs:
        model.set_adapter(BuggyAdapter(raw_data))
    return model, [record.getMessage() for record in logs.records]


def test_adapter_returning_invalid_attribute():
    # 3 columns of editable flags for 2 columns of values
    model, errors = model_logged_errors({'values': [[1, 2],
                                                    [3, 4]],
                                         'editable': [[True, True, True]]})
    assert errors == ["BuggyAdapter returned 'editable' with shape (1, 3), which does not "
                      "broadcast to the shape of its values (2, 2): ignoring it"]
    # the values are displayed anyway (but not editable)
    displayed_text = model.data(model.index(1, 1), Qt.DisplayRole)
    assert displayed_text == '4'
    assert not model.flags(model.index(0, 0)) & Qt.ItemIsEditable


def test_adapter_returning_invalid_values():
    # rows of different lengths
    model, errors = model_logged_errors({'values': [[1, 2],
                                                    [3]]})
    assert errors == ["BuggyAdapter returned values which are not a 2D sequence (with rows "
                      "of the same length): ignoring them"]
    assert model.columnCount() == 0
