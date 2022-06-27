from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtCore import QPoint, Qt


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


class FilterMenu(QtWidgets.QMenu):
    activate = QtCore.Signal(int)
    checkedItemsChanged = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._list_view = QtWidgets.QListView(parent)
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

        action = QtWidgets.QWidgetAction(self)
        action.setDefaultWidget(self._list_view)
        self.addAction(action)
        self.installEventFilter(self)
        self._list_view.installEventFilter(self)
        self._list_view.window().installEventFilter(self)

        model.itemChanged.connect(self.on_model_item_changed)
        self._list_view.pressed.connect(self.on_list_view_pressed)
        self.activate.connect(self.on_activate)

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
        self.checkedItemsChanged.emit(checked_indices)

    def select_offset(self, offset):
        """offset: 1 for next, -1 for previous"""

        model = self._model
        model.blockSignals(True)
        indices_checked = [i for i, item in enumerate(model) if item.checked]
        first_checked = indices_checked[0]
        # check first_checked + offset, uncheck the rest
        to_check = first_checked + offset

        # wrap around
        to_check = to_check if to_check < len(model) else 1
        to_check = to_check if to_check > 0 else len(model) - 1

        is_checked = ["partial"] + [i == to_check for i in range(1, len(model))]
        for checked, item in zip(is_checked, model):
            item.checked = checked
        model.blockSignals(False)
        self.checkedItemsChanged.emit([to_check - 1])

    def addItem(self, text):
        item = StandardItem(text)
        # not editable
        item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        item.checked = True
        self._model.appendRow(item)

    def addItems(self, items):
        for item in items:
            self.addItem(item)

    def eventFilter(self, obj, event):
        event_type = event.type()

        if event_type == QtCore.QEvent.KeyRelease:
            key = event.key()

            # tab key closes the popup
            if obj == self._list_view.window() and key == Qt.Key_Tab:
                self.hide()

            # return key activates *one* item and closes the popup
            # first time the key is sent to the menu, afterwards to
            # list_view
            elif obj == self._list_view and key in (Qt.Key_Enter, Qt.Key_Return):
                self.activate.emit(self._list_view.currentIndex().row())
                self.hide()
                return True

        return False


class FilterComboBox(QtWidgets.QToolButton):
    checkedItemsChanged = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("(no filter)")
        # QtGui.QToolButton.InstantPopup would be slightly less work (the
        # whole button works by default, instead of only the arrow) but it is
        # uglier
        self.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)

        menu = FilterMenu(self)
        self.setMenu(menu)
        self._menu = menu
        menu.checkedItemsChanged.connect(self.on_checked_items_changed)
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
        self.checkedItemsChanged.emit(indices_checked)

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
            elif obj == self and key in (Qt.Key_Enter, Qt.Key_Return):
                self._menu.activate.emit(self._list_view.currentIndex().row())
                self._menu.hide()
                return True

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
