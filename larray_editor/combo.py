from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtCore import QPoint, Qt

from larray_editor.utils import create_action, _


class StandardItemModelIterator:
    def __init__(self, model):
        self.model = model
        self.pos = 0

    def __next__(self):
        if self.pos < self.model.rowCount():
            item = self.model.item(self.pos)
            self.pos += 1
            return item
        else:
            raise StopIteration
    next = __next__

    def __iter__(self):
        return self


class SequenceStandardItemModel(QtGui.QStandardItemModel):
    """
    an iterable and indexable StandardItemModel
    """
    def __iter__(self):
        return StandardItemModelIterator(self)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.rowCount()
            if step is None:
                step = 1
            return [self.item(i) for i in range(start, stop, step)]
        else:
            if key >= self.rowCount():
                raise IndexError(f"index {key} is out of range")
            return self.item(key)

    def __len__(self):
        return self.rowCount()


class StandardItem(QtGui.QStandardItem):
    def __init__(self, value):
        super().__init__(value)

    def get_checked(self):
        return self.checkState() == Qt.Checked

    def set_checked(self, value):
        if isinstance(value, bool):
            qtvalue = (Qt.Unchecked, Qt.Checked)[value]
        else:
            qtvalue = Qt.PartiallyChecked
        self.setCheckState(qtvalue)
    checked = property(get_checked, set_checked)


class CombinedSortFilterMenu(QtWidgets.QMenu):
    activate = QtCore.Signal(int)
    checked_items_changed = QtCore.Signal(list)
    sort_signal = QtCore.Signal(bool)  # bool argument is for ascending

    def __init__(self, parent=None,
                 sortable: bool = False,
                 sort_direction: str = 'unsorted',
                 filtrable=False):
        super().__init__(parent)

        self._model, self._list_view = None, None

        if sortable:
            self.addAction(create_action(self, _('Sort A-Z'),
                                         triggered=lambda: self.sort_signal.emit(True),
                                         checkable=True,
                                         checked=sort_direction == 'ascending'))
            self.addAction(create_action(self, _('Sort Z-A'),
                                         triggered=lambda: self.sort_signal.emit(False),
                                         checkable=True,
                                         checked=sort_direction == 'descending'))
            if filtrable:
                self.addSeparator()

        if filtrable:
            self.setup_list_view()

        self.installEventFilter(self)
        self.activate.connect(self.on_activate)

    def setup_list_view(self):
        # search_widget = QtWidgets.QLineEdit()
        # search_widget.setPlaceholderText('Search')
        # self.add_action_widget(search_widget)

        self._list_view = QtWidgets.QListView(self)
        self._list_view.setFrameStyle(0)
        model = SequenceStandardItemModel()
        self._list_view.setModel(model)
        self._model = model
        self.addItem("(select all)")
        try:
            model[0].setTristate(True)
        except AttributeError:
            # this is the new name for qt6+
            model[0].setUserTristate(True)
        self._list_view.installEventFilter(self)
        self._list_view.window().installEventFilter(self)
        model.itemChanged.connect(self.on_model_item_changed)
        self._list_view.pressed.connect(self.on_list_view_pressed)

        # filters_layout = QtWidgets.QVBoxLayout(parent)
        # filters_layout.addWidget(QtWidgets.QLabel("Filters"))
        # filters_layout.addWidget(self._list_view)
        self.add_action_widget(self._list_view)

    def add_action_widget(self, action_widget):
        if isinstance(action_widget, QtWidgets.QLayout):
            # you cant add a layout directly in an action, so we have to wrap it in a widget
            widget = QtWidgets.QWidget()
            widget.setLayout(action_widget)
            action_widget = widget
        widget_action = QtWidgets.QWidgetAction(self)
        widget_action.setDefaultWidget(action_widget)
        self.addAction(widget_action)

    def on_list_view_pressed(self, index):
        item = self._model.itemFromIndex(index)
        # item is None when the button has not been used yet (and this is
        # triggered via enter)
        if item is not None:
            item.checked = not item.checked

    def on_activate(self, row):
        target_item = self._model[row]
        for item in self._model[1:]:
            item.checked = item is target_item

    def on_model_item_changed(self, item):
        model = self._model
        model.blockSignals(True)
        if item.index().row() == 0:
            # (un)check first => (un)check others
            for other in model[1:]:
                other.checked = item.checked

        items_checked = [item for item in model[1:] if item.checked]
        num_checked = len(items_checked)

        if num_checked == 0 or num_checked == len(model) - 1:
            model[0].checked = bool(num_checked)
        elif num_checked == 1:
            model[0].checked = 'partial'
        else:
            model[0].checked = 'partial'
        model.blockSignals(False)
        checked_indices = [i for i, item in enumerate(model[1:]) if item.checked]
        self.checked_items_changed.emit(checked_indices)

    # function is called to implement wheel scrolling (select prev/next label)
    def select_offset(self, offset):
        """offset: -1 for previous, 1 for next"""
        assert offset in {-1, 1}
        model = self._model
        model.blockSignals(True)
        # Remember the "(select all)" label shifts all indices by one
        indices_checked = [i for i, item in enumerate(model) if item.checked]
        if indices_checked:
            first_checked = indices_checked[0]
        else:
            # if no label is checked, act like "(select all)" was checked
            # (i.e. we will select the first real label)
            first_checked = 0
        # check first_checked + offset, uncheck the rest
        to_check = first_checked + offset

        # wrap around (index 0 is reserved for "(select all)")
        to_check = to_check if to_check < len(model) else 1
        to_check = to_check if to_check > 0 else len(model) - 1

        model[0].checked = "partial"
        for i, item in enumerate(model[1:], start=1):
            item.checked = i == to_check
        model.blockSignals(False)
        self.checked_items_changed.emit([to_check - 1])

    def addItem(self, text, checked=True):
        item = StandardItem(text)
        # not editable
        item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        item.checked = checked
        self._model.appendRow(item)

    def addItems(self, items, items_checked=None):
        if items_checked is None:
            for item_label in items:
                self.addItem(item_label)
        else:
            assert 0 <= len(items_checked) <= len(items)
            checked_indices_set = set(items_checked)
            for idx, item_label in enumerate(items):
                self.addItem(item_label, idx in checked_indices_set)

    def eventFilter(self, obj, event):
        event_type = event.type()

        if event_type == QtCore.QEvent.KeyRelease:
            key = event.key()

            # tab key closes the popup
            if obj == self._list_view.window() and key == QtCore.Qt.Key_Tab:
                self.hide()

            # return key activates *one* item and closes the popup
            # first time the key is sent to the menu, afterwards to
            # list_view
            elif (obj == self._list_view and
                          key in (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return)):
                self.activate.emit(self._list_view.currentIndex().row())
                self.hide()
                return True

        return False


class FilterComboBox(QtWidgets.QToolButton):
    checked_items_changed = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("(no filter)")
        # QtGui.QToolButton.InstantPopup would be slightly less work (the
        # whole button works by default, instead of only the arrow) but it is
        # uglier
        self.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)

        menu = CombinedSortFilterMenu(self, filtrable=True)
        self.setMenu(menu)
        self._menu = menu
        menu.checked_items_changed.connect(self.on_checked_items_changed)
        self.installEventFilter(self)

    def on_checked_items_changed(self, indices_checked):
        num_checked = len(indices_checked)
        model = self._menu._model
        if num_checked == 0 or num_checked == len(model) - 1:
            self.setText("(no filter)")
        elif num_checked == 1:
            self.setText(model[indices_checked[0] + 1].text())
        else:
            self.setText("multi")
        self.checked_items_changed.emit(indices_checked)

    def addItem(self, text):
        self._menu.addItem(text)

    def addItems(self, items):
        self._menu.addItems(items)

    def eventFilter(self, obj, event):
        event_type = event.type()

        # this is not enabled because it causes all kind of troubles
        # if event_type == QtCore.QEvent.KeyPress:
        #     key = event.key()
        #
        #     # allow opening the popup via enter/return
        #     if obj == self and key in (Qt.Key_Return, Qt.Key_Enter):
        #         self.showMenu()
        #         return True

        if event_type == QtCore.QEvent.KeyRelease:
            key = event.key()

            # allow opening the popup with up/down
            if obj == self and key in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Space):
                self.showMenu()
                return True

            # return key activates *one* item and closes the popup
            # first time the key is sent to self, afterwards to list_view
            # elif obj == self and key in (Qt.Key_Enter, Qt.Key_Return):
            #     print(f'FilterComboBox.eventFilter')
            #     # this cannot work (there is no _list_view attribute)
            #     # probably meant as self._menu._list_view BUT
            #     # this case currently does not seem to happen anyway
            #     # I am not removing this code entirely because the
            #     # combo does not seem to get focus which could explain
            #     # why this is never reached
            #     current_index = self._list_view.currentIndex().row()
            #     print(f'current_index={current_index}')
            #     self._menu.activate.emit(current_index)
            #     self._menu.hide()
            #     return True

        if event_type == QtCore.QEvent.MouseButtonRelease:
            # clicking anywhere (not just arrow) on the button shows the popup
            if obj == self:
                self.showMenu()

        return False

    def wheelEvent(self, event):
        delta = event.angleDelta()
        assert isinstance(delta, QPoint)
        offset = 1 if delta.y() < 0 else -1
        self._menu.select_offset(offset)


if __name__ == '__main__':
    import sys

    class TestDialog(QtWidgets.QDialog):
        def __init__(self):
            super().__init__()
            layout = QtWidgets.QVBoxLayout()
            self.setLayout(layout)

            combo = FilterComboBox(self)
            for i in range(20):
                combo.addItem(f'Item {i}')
            layout.addWidget(combo)

    app = QtWidgets.QApplication(sys.argv)
    dialog = TestDialog()
    dialog.resize(200, 200)
    dialog.show()
    sys.exit(app.exec_())
