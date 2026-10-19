import os
from unittest import mock

import pytest
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QApplication

from larray_editor.combo import CombinedSortFilterMenu


# TODO: temporary, remove this once these tests are known to work on CI, which runs without
#       display (they should, using the offscreen Qt platform, see conftest.py)
pytestmark = pytest.mark.skipif(bool(os.environ.get('CI')),
                                reason="Qt tests are temporarily skipped on CI")

# the Qt application must stay referenced, otherwise it is garbage collected
qt_app = None


def create_menu(**kwargs):
    """Create a sort/filter menu (and the Qt application if needed)"""
    global qt_app
    qt_app = QApplication.instance() or QApplication([])
    return CombinedSortFilterMenu(**kwargs)


def test_key_release_in_sort_only_menu():
    menu = create_menu(sortable=True)
    key_release = QKeyEvent(QEvent.KeyRelease, Qt.Key_Down, Qt.NoModifier)
    event_filtered = menu.eventFilter(menu, key_release)
    assert not event_filtered

    # tab closes the menu, like menus with filters
    tab_release = QKeyEvent(QEvent.KeyRelease, Qt.Key_Tab, Qt.NoModifier)
    with mock.patch.object(menu, 'hide') as hide:
        menu.eventFilter(menu, tab_release)
    hide.assert_called_once()


def test_activate_filter_item():
    menu = create_menu(filtrable=True)
    menu.addItems(['a', 'b', 'c'])
    emitted = []
    menu.checked_items_changed.connect(emitted.append)

    # activating (e.g. pressing Enter on) an item checks only that item, and the filter is
    # updated once
    menu.on_activate(2)
    assert emitted == [[1]]
