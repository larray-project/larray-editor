# -*- coding: utf-8 -*-
#
# Copyright © 2009-2012 Pierre Raybaut
# Copyright © 2015-2025 Gaëtan de Menten
# Licensed under the terms of the MIT License

# based on
# github.com/spyder-ide/spyder/blob/master/spyderlib/widgets/arrayeditor.py

"""
Array Editor Dialog based on Qt
"""

# pylint: disable=C0103
# pylint: disable=R0903
# pylint: disable=R0911
# pylint: disable=R0201

# Note that the canonical way to implement filters in a TableView would
# be to use a QSortFilterProxyModel. In this case, we would need to reimplement
# its filterAcceptsColumn and filterAcceptsRow methods. The problem is that
# it does not seem to be really designed for very large arrays and it would
# probably be too slow on those (I have read quite a few people complaining
# about speed issues with those) possibly because it suppose you have the whole
# array in your model. It would also probably not play well with the
# partial/progressive load we have currently implemented.

# TODO:
# * make it more obvious one can drag & drop axes names to reorder axes
#   http://ux.stackexchange.com/questions/34158/
#       how-to-make-it-obvious-that-you-can-drag-things-that-you-normally-cant
# * document paint speed experiments
# * filter on headers. In fact this is not a good idea, because that prevents
#   selecting whole columns, which is handy. So a separate row for headers,
#   like in Excel seems better.
# * tooltip on header with current filter

# * selection change -> select headers too
# * nicer error on plot with more than one row/column
#   OR
# * plotting a subset should probably (to think) go via LArray/pandas objects
#   so that I have the headers info in the plots (and do not have to deal with
#   them manually)
#   > ideally, I would like to keep this generic (not LArray-specific)
# ? automatic change digits on resize column
#   => different format per column, which is problematic UI-wise
# * keyboard shortcut for filter each dim
# * tab in a filter combo, brings up next filter combo
# ? custom delegates for each type (spinner for int, checkbox for bool, ...)
# ? "light" headers (do not repeat the same header several times (on the screen)
#   it would be nicer but I am not sure it is a good idea because with many
#   dimensions, you can no longer see the current label for the first
#   dimension(s) if you scroll down a bit. This is solvable if, instead
#   of only the first line ever corresponding to the label displaying it,
#   I could make it so that it is the first line displayable on the screen
#   which gets it. It would be a bit less nice because of strange artifacts
#   when scrolling, but would be more useful. The beauty problem could be
#   solved later too via fading or something like that, but probably not
#   worth it for a while.

import math
from pathlib import Path

import numpy as np

from qtpy import QtCore
from qtpy.QtCore import (Qt, QPoint, QItemSelection, QItemSelectionModel,
                         Signal, QSize, QModelIndex, QTimer)
from qtpy.QtGui import (QDoubleValidator, QIntValidator, QKeySequence, QFontMetrics, QCursor, QPixmap, QPainter, QIcon,
                        QWheelEvent, QMouseEvent)
from qtpy.QtWidgets import (QApplication, QTableView, QItemDelegate, QLineEdit, QCheckBox,
                            QMessageBox, QMenu, QLabel, QSpinBox, QWidget, QToolTip, QShortcut, QScrollBar,
                            QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy, QFrame, QComboBox,
                            QStyleOptionViewItem, QPushButton)

from larray_editor.utils import (keybinding, create_action, clear_layout, get_default_font,
                                 is_number_dtype, is_float_dtype, _,
                                 LinearGradient, logger, cached_property, data_frac_digits,
                                 num_int_digits)
from larray_editor.arrayadapter import (get_adapter, get_adapter_creator,
                                        AbstractAdapter, MAX_FILTER_OPTIONS)
from larray_editor.arraymodel import (HLabelsArrayModel, VLabelsArrayModel, LabelsArrayModel,
                                      AxesArrayModel, DataArrayModel)
from larray_editor.combo import FilterComboBox, CombinedSortFilterMenu

MORE_OPTIONS_NOT_SHOWN = "<more options not shown>"

# mime-type we use when drag and dropping axes (x- prefix is for unregistered
# types)
LARRAY_AXIS_INDEX_DRAG_AND_DROP_MIMETYPE = "application/x-larray-axis-index"


def display_selection(selection: QtCore.QItemSelection):
    return ', '.join(f"<{idx.row()}, {idx.column()}>" for idx in selection.indexes())


def clip(value, minimum, maximum):
    if value < minimum:
        return minimum
    elif value > maximum:
        return maximum
    else:
        return value


# XXX: define Enum instead ?
TOP, BOTTOM = 0, 1
LEFT, RIGHT = 0, 1
MIN_COLUMN_WIDTH = 30
MAX_COLUMN_WIDTH = 800
DEFAULT_COLUMN_WIDTH = 64
DEFAULT_ROW_HEIGHT = 20


class FilterBar(QWidget):
    def __init__(self, array_widget):
        super().__init__()
        # we are not passing array_widget as parent for QHBoxLayout because
        # we could have the filterbar outside the widget
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.array_widget = array_widget

        # See https://www.pythonguis.com/faq/pyqt-drag-drop-widgets/
        # and https://zetcode.com/pyqt6/dragdrop/
        self.setAcceptDrops(True)
        self.drag_label = None
        self.drag_start_pos = None

    def reset_to_defaults(self):
        layout = self.layout()
        clear_layout(layout)
        data_adapter = self.array_widget.data_adapter
        if data_adapter is None:
            return
        assert isinstance(data_adapter, AbstractAdapter), \
            f"unexpected data_adapter type: {type(data_adapter)}"
        filter_names = data_adapter.get_filter_names()
        # size > 0 to avoid arrays with length 0 axes and len(axes) > 0 to avoid scalars (scalar.size == 1)
        if filter_names: #self.data_adapter.size > 0 and len(filters) > 0:
            layout.addWidget(QLabel(_("Filters")))
            for filter_idx, filter_name in enumerate(filter_names):
                layout.addWidget(QLabel(filter_name))
                filter_labels = data_adapter.get_filter_options(filter_idx)
                # FIXME: on very large axes, this is getting too slow. Ideally the combobox should use a model which
                #        only fetch labels when they are needed to be displayed
                #        this needs a whole new widget though
                if len(filter_labels) < 10000:
                    layout.addWidget(self.create_filter_combo(filter_idx, filter_labels))
                else:
                    layout.addWidget(QLabel("too big to be filtered"))
            layout.addStretch()

    def create_filter_combo(self, filter_idx, filter_labels):
        def filter_changed(checked_items):
            self.change_filter(filter_idx, checked_items)

        combo = FilterComboBox(self)
        combo.addItems([str(label) for label in filter_labels])
        combo.checked_items_changed.connect(filter_changed)
        return combo

    def change_filter(self, filter_idx, indices):
        logger.debug(f"FilterBar.change_filter({filter_idx}, {indices})")
        # FIXME: the method can be called from the outside, and in that case
        #        the combos checked items need be synchronized too
        array_widget = self.array_widget
        data_adapter = array_widget.data_adapter
        vscrollbar: ScrollBar = array_widget.vscrollbar
        hscrollbar: ScrollBar = array_widget.hscrollbar
        old_v_pos = vscrollbar.value()
        old_h_pos = hscrollbar.value()
        old_nrows, old_ncols = data_adapter.shape2d()
        data_adapter.update_filter(filter_idx, indices)
        data_adapter._current_sort = []
        # TODO: this does too much work (it sets the adapters even
        #       if those do not change and sets v_offset/h_offset to 0 when we
        #       do not *always* want to do so) and maybe too little
        #       (update_range should probably be done elsewhere)
        #       this also reset() each model.
        #       For DataArrayModel it causes an extra (compared to the one
        #       below) update_range (via the modelReset signal)
        array_widget._set_models_adapter()
        new_nrows, new_ncols = data_adapter.shape2d()
        hscrollbar.update_range()
        vscrollbar.update_range()
        array_widget.update_cell_sizes_from_content()
        if old_v_pos == 0 and old_h_pos == 0:
            # if the old values were already 0, visible_v/hscroll_changed will
            # not be triggered and update_*_column_widths has no chance to run
            # unless we call them explicitly
            assert isinstance(array_widget, ArrayEditorWidget)
            array_widget.update_cell_sizes_from_content()
        else:
            # TODO: would be nice to implement some clever positioning algorithm
            #       here when new_X != old_X so that the visible rows stay visible.
            #       Currently, this does not change the scrollbar value at all if
            #       the old value fits in the new range. When changing from one
            #       specific label to another of an larray, this does not change
            #       the shape of the result and is thus what we want but there
            #       are cases where we could do better.
            # TODO: the setValue(0) should not be necessary in the case of
            #       new_nrows == old_nrows but it is currently because
            #       v/h_offset is set to 0 by the call to _set_models_adapter
            #       above but the scrollbar values do not change, so
            #       setValue(old_v_pos) does not trigger a valueChanged signal,
            #       and thus the v/h_offset is not set back to its old value
            #       if we don't first change the scrollbar values
            vscrollbar.setValue(0)
            hscrollbar.setValue(0)
            # if the old value was already at 0, we do not need to set it again
            if new_nrows == old_nrows and old_v_pos != 0:
                vscrollbar.setValue(old_v_pos)
            if new_ncols == old_ncols and old_h_pos != 0:
                hscrollbar.setValue(old_h_pos)

    # Check for left button mouse press events on axis labels
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        click_pos = event.pos()
        child = self.childAt(click_pos)
        assert self.drag_label is None
        if isinstance(child, QLabel) and child.text() != "Filters":
            self.drag_label = child
            self.drag_start_pos = click_pos

    # If we release the left button before we moved the mouse enough to
    # trigger the "real" dragging sequence (see mouveMoveEvent), we need to
    # forget the drag_label and drag_start_pos
    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self.drag_label = None
        self.drag_start_pos = None

    # Mouse move events will occur only when a mouse button is pressed down,
    # unless mouse tracking has been enabled with QWidget.setMouseTracking()
    def mouseMoveEvent(self, event):
        # We did not click on an axis label yet
        drag_label = self.drag_label
        if drag_label is None:
            return

        # We do not check the event button. The left button should still be
        # pressed but event.button() will always be NoButton: "If the event type
        # is MouseMove, the appropriate button for this event is Qt::NoButton"

        # We are too close to where we initially clicked
        drag_delta = event.pos() - self.drag_start_pos
        if drag_delta.manhattanLength() < QApplication.startDragDistance():
            return

        from qtpy.QtCore import QMimeData, QByteArray
        from qtpy.QtGui import QDrag

        axis_index = self.layout().indexOf(drag_label) // 2

        mimeData = QMimeData()
        mimeData.setData(LARRAY_AXIS_INDEX_DRAG_AND_DROP_MIMETYPE,
                         QByteArray.number(axis_index))
        pixmap = QPixmap(drag_label.size())
        drag_label.render(pixmap)

        # We will initiate a real dragging sequence, we don't need these anymore
        self.drag_label = None
        self.drag_start_pos = None

        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setPixmap(pixmap)
        drag.setHotSpot(drag_delta)
        drag.exec_(Qt.MoveAction)

    # Tell whether the filter bar is an acceptable target for a particular
    # dragging event (which could come from another app)
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(LARRAY_AXIS_INDEX_DRAG_AND_DROP_MIMETYPE):
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()

    # Inside the filter bar, inform Qt whether some particular position
    # is a good final target or not
    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(LARRAY_AXIS_INDEX_DRAG_AND_DROP_MIMETYPE):
            child = self.childAt(event.pos())
            if isinstance(child, QLabel) and child.text() != "Filters":
                event.setDropAction(Qt.MoveAction)
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    # If the user dropped on a valid target, we need to handle the event
    def dropEvent(self, event):
        mime_data = event.mimeData()
        if mime_data.hasFormat(LARRAY_AXIS_INDEX_DRAG_AND_DROP_MIMETYPE):
            old_index_byte_array = mime_data.data(LARRAY_AXIS_INDEX_DRAG_AND_DROP_MIMETYPE)
            old_index, success = old_index_byte_array.toInt()
            child = self.childAt(event.pos())
            new_index = self.layout().indexOf(child) // 2
            data_adapter = self.array_widget.data_adapter
            data, attributes = data_adapter.move_axis(data_adapter.data,
                                                      data_adapter.attributes,
                                                      old_index,
                                                      new_index)
            self.array_widget.set_data(data, attributes)
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()


class BackButtonBar(QWidget):
    def __init__(self, array_widget):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        button = QPushButton('Back')
        button.clicked.connect(self.on_clicked)
        layout.addWidget(button)
        self.array_widget = array_widget
        self._back_data = []
        self._back_data_adapters = []
        layout.addStretch()
        self.hide()

    def add_back(self, data, data_adapter):
        self._back_data.append(data)
        # We need to keep the data_adapter around because some resource
        # are created in the adapter (e.g. duckdb connection when viewing a
        # .ddb file) and if the adapter is garbage collected, the resource
        # is deleted (e.g. the duckdb connection dies - contrary to other libs,
        # a duckdb table object does not keep the connection alive)
        self._back_data_adapters.append(data_adapter)
        if not self.isVisible():
            self.show()

    def clear(self):
        for adapter in self._back_data_adapters[::-1]:
            self._close_adapter(adapter)

        self._back_data_adapters = []
        self._back_data = []
        self.hide()

    @staticmethod
    def _close_adapter(adapter):
        clsname = type(adapter).__name__
        logger.debug(f"closing data adapter ({clsname})")
        adapter.close()

    def on_clicked(self):
        if not len(self._back_data):
            logger.warn("Back button has no target to go to")
            return
        target_data = self._back_data.pop()
        data_adapter = self._back_data_adapters.pop()
        if not len(self._back_data):
            self.hide()
        array_widget: ArrayEditorWidget = self.array_widget
        # We are not using array_widget.set_data(target_data) so that we can
        # reuse the same data_adapter instead of recreating a new one
        array_widget.data = target_data
        array_widget.set_data_adapter(data_adapter, frac_digits=None)


class AbstractView(QTableView):
    """Abstract view class"""
    def __init__(self, parent, model, hpos, vpos):
        assert isinstance(parent, ArrayEditorWidget)
        QTableView.__init__(self, parent)

        # set model
        self.setModel(model)

        # set position
        if hpos not in {LEFT, RIGHT}:
            raise TypeError(f"Value of hpos must be {LEFT} or {RIGHT}")
        self.hpos = hpos
        if vpos not in {TOP, BOTTOM}:
            raise TypeError(f"Value of vpos must be {TOP} or {BOTTOM}")
        self.vpos = vpos
        self.first_selection_corner = None
        # handling a second selection corner is necessary to implement the
        # "select entire row/column" functionality because in that case the
        # second corner is not necessarily in the viewport, but it is a real
        # cell (i.e. the coordinates are inclusive)
        self.second_selection_corner = None

        # set selection mode
        self.setSelectionMode(QTableView.ContiguousSelection)

        # prepare headers + cells size
        self.horizontalHeader().setFrameStyle(QFrame.NoFrame)
        self.verticalHeader().setFrameStyle(QFrame.NoFrame)
        self.set_default_size()
        # hide horizontal/vertical headers
        if hpos == RIGHT:
            self.verticalHeader().hide()
        if vpos == BOTTOM:
            self.horizontalHeader().hide()

        # XXX: this might help if we want the widget to be focusable using "tab"
        # self.setFocusPolicy(Qt.StrongFocus)

        # These 4 lines are only useful for debugging
        # hscrollbar = self.horizontalScrollBar()
        # hscrollbar.valueChanged.connect(self.on_horizontal_scroll_changed)
        # vscrollbar = self.verticalScrollBar()
        # vscrollbar.valueChanged.connect(self.on_vertical_scroll_changed)

        # Hide scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # update geometry
        if not (hpos == RIGHT and vpos == BOTTOM):
            self.model().modelReset.connect(self.updateGeometry)
            self.horizontalHeader().sectionResized.connect(self.updateGeometry)
            self.verticalHeader().sectionResized.connect(self.updateGeometry)

    # def on_vertical_scroll_changed(self, value):
    #     log_caller()
    #     print(f"hidden vscroll on {self.__class__.__name__} changed to {value}")

    # def on_horizontal_scroll_changed(self, value):
    #     log_caller()
    #     print(f"hidden hscroll on {self.__class__.__name__} changed to {value}")

    # def selectionChanged(self, selected: QtCore.QItemSelection, deselected: QtCore.QItemSelection) -> None:
    #     super().selectionChanged(selected, deselected)
    #     print(f"selectionChanged:\n"
    #           f"   -> selected({display_selection(selected)}),\n"
    #           f"   -> deselected({display_selection(deselected)})")

    def reset_to_defaults(self):
        """
        reset widget to initial state (when the ArrayEditorWidget is switching
        from one object to another)
        """
        self.set_default_size()
        self.first_selection_corner = None
        self.second_selection_corner = None

    def set_default_size(self):
        # logger.debug(f"{self.__class__.__name__}.set_default_size()")

        # make the grid a bit more compact
        horizontal_header = self.horizontalHeader()
        horizontal_header.blockSignals(True)
        horizontal_header.setDefaultSectionSize(DEFAULT_COLUMN_WIDTH)

        if horizontal_header.sectionSize(0) != DEFAULT_COLUMN_WIDTH:
            # Explicitly set all columns to the default width to override any
            # custom sizes
            for col in range(self.model().columnCount()):
                self.setColumnWidth(col, DEFAULT_COLUMN_WIDTH)
        horizontal_header.blockSignals(False)

        self.verticalHeader().setDefaultSectionSize(DEFAULT_ROW_HEIGHT)
        if self.vpos == TOP:
            horizontal_header.setFixedHeight(10)
        if self.hpos == LEFT:
            self.verticalHeader().setFixedWidth(10)

    # We need to have this here (in AbstractView) and not only on DataView, so that we
    # catch them for vlabels too. For axes and hlabels, it is a bit of a weird
    # behavior since they are not affected themselves but that is really a nitpick
    # Also, overriding the general event() method for this does not work as it is
    # handled behind us (by the ScrollArea I assume) and we do not even see the event
    # unless we are at the buffer boundary.
    def wheelEvent(self, event: QWheelEvent):
        """Catch wheel events and send them to the corresponding visible scrollbar"""
        delta = event.angleDelta()
        logger.debug(f"wheelEvent on {self.__class__.__name__} ({delta})")
        editor_widget = self.parent().parent()
        if delta.x() != 0:
            editor_widget.hscrollbar.wheelEvent(event)
        if delta.y() != 0:
            editor_widget.vscrollbar.wheelEvent(event)
        event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        if key in {Qt.Key_Home, Qt.Key_End, Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right,
                   Qt.Key_PageUp, Qt.Key_PageDown}:
            event.accept()
            self.navigate_key_event(event)
        else:
            QTableView.keyPressEvent(self, event)

    def navigate_key_event(self, event):
        logger.debug("")
        logger.debug("navigate")
        logger.debug("========")
        model = self.model()
        widget = self.parent().parent()
        assert isinstance(widget, ArrayEditorWidget)

        event_modifiers = event.modifiers()
        event_key = event.key()
        if event_modifiers & Qt.ShiftModifier:
            # remove shift from modifiers so the Ctrl+Key combos are still detected
            event_modifiers ^= Qt.ShiftModifier
            shift = True
        else:
            shift = False

        try:
            # qt6
            modifiers_value = event_modifiers.value
        except AttributeError:
            # qt5
            modifiers_value = event_modifiers
        keyseq = QKeySequence(modifiers_value | event_key)
        page_step = self.verticalScrollBar().pageStep()
        cursor_global_pos = self.get_cursor_global_pos()
        if cursor_global_pos is None:
            cursor_global_v_pos, cursor_global_h_pos = 0, 0
            logger.debug("No previous cursor position: using 0, 0")
        else:
            cursor_global_v_pos, cursor_global_h_pos = cursor_global_pos
            logger.debug(f"old global cursor {cursor_global_v_pos} {cursor_global_h_pos}")

        # TODO: for some adapter shape2 is not reliable (it is a best guess), we should make sure we gracefully handle
        #       wrong info
        total_v_size, total_h_size = model.adapter.shape2d()
        key2delta = {
            Qt.Key_Home: (0, -cursor_global_h_pos),
            Qt.Key_End: (0, total_h_size - cursor_global_h_pos - 1),
            Qt.Key_Up: (-1, 0),
            Qt.Key_Down: (1, 0),
            Qt.Key_Left: (0, -1),
            Qt.Key_Right: (0, 1),
            Qt.Key_PageUp: (-page_step, 0),
            Qt.Key_PageDown: (page_step, 0),
        }

        # Ctrl+arrow does not mean anything by default, so dispatching does not help
        # TODO: use another dict for this. dict[keyseq] does not work even if keyseq == key works.
        # Using a different dict and checking the modifier explicitly should work.
        # Or maybe getting the string representation of the keyseq is possible too.
        # TODO: it might be simpler to set the cursor_global_pos values directly rather than using delta
        if keyseq == "Ctrl+Home":
            v_delta, h_delta = (-cursor_global_v_pos, -cursor_global_h_pos)
        elif keyseq == "Ctrl+End":
            v_delta, h_delta = (total_v_size - cursor_global_v_pos - 1, total_h_size - cursor_global_h_pos - 1)
        elif keyseq == "Ctrl+Left":
            v_delta, h_delta = (0, -cursor_global_h_pos)
        elif keyseq == "Ctrl+Right":
            v_delta, h_delta = (0, total_h_size - cursor_global_h_pos - 1)
        elif keyseq == "Ctrl+Up":
            v_delta, h_delta = (-cursor_global_v_pos, 0)
        elif keyseq == "Ctrl+Down":
            v_delta, h_delta = (total_v_size - cursor_global_v_pos - 1, 0)
        else:
            v_delta, h_delta = key2delta[event_key]

        # TODO: internal scroll => change value of visible scrollbar (or avoid internal scroll)
        cursor_new_global_v_pos = clip(cursor_global_v_pos + v_delta, 0, total_v_size - 1)
        cursor_new_global_h_pos = clip(cursor_global_h_pos + h_delta, 0, total_h_size - 1)
        logger.debug(f"new global cursor {cursor_new_global_v_pos} {cursor_new_global_h_pos}")

        self.scroll_to_global_pos(cursor_new_global_v_pos, cursor_new_global_h_pos)

        new_v_posinbuffer = cursor_new_global_v_pos - model.v_offset
        new_h_posinbuffer = cursor_new_global_h_pos - model.h_offset

        local_cursor_index = model.index(new_v_posinbuffer, new_h_posinbuffer)
        if shift:
            if self.first_selection_corner is None:
                # This can happen when using navigation keys before
                # selecting any cell using the mouse (but after getting focus
                # on the widget which can be done at least by clicking inside
                # the widget area but outside "valid" cells)
                self.first_selection_corner = (cursor_global_v_pos, cursor_global_h_pos)
            self.second_selection_corner = cursor_new_global_v_pos, cursor_new_global_h_pos
            selection_v_pos1, selection_h_pos1 = self.first_selection_corner
            selection_v_pos2, selection_h_pos2 = self.second_selection_corner
            row_min = min(selection_v_pos1, selection_v_pos2)
            row_max = max(selection_v_pos1, selection_v_pos2)
            col_min = min(selection_h_pos1, selection_h_pos2)
            col_max = max(selection_h_pos1, selection_h_pos2)

            selection_model = self.selectionModel()
            selection_model.setCurrentIndex(local_cursor_index, QItemSelectionModel.Current)
            # we need to clip local coordinates in case the selection corners are outside the viewport
            local_top = max(row_min - model.v_offset, 0)
            local_left = max(col_min - model.h_offset, 0)
            local_bottom = min(row_max - model.v_offset, model.nrows - 1)
            local_right = min(col_max - model.h_offset, model.ncols - 1)
            selection = QItemSelection(model.index(local_top, local_left),
                                       model.index(local_bottom, local_right))
            selection_model.select(selection, QItemSelectionModel.ClearAndSelect)
        else:
            self.first_selection_corner = cursor_new_global_v_pos, cursor_new_global_h_pos
            self.second_selection_corner = cursor_new_global_v_pos, cursor_new_global_h_pos
            self.setCurrentIndex(local_cursor_index)
        logger.debug(f"after navigate_key_event: {self.first_selection_corner=} "
                     f"{self.second_selection_corner=}")

    # after we drop support for Python < 3.10, we should use:
    # def get_cursor_global_pos(self) -> tuple[int, int] | None:
    def get_cursor_global_pos(self):
        model = self.model()
        current_index = self.currentIndex()
        if not current_index.isValid():
            return None
        v_posinbuffer = current_index.row()
        h_posinbuffer = current_index.column()
        assert v_posinbuffer >= 0
        assert h_posinbuffer >= 0
        cursor_global_v_pos = model.v_offset + v_posinbuffer
        cursor_global_h_pos = model.h_offset + h_posinbuffer
        return cursor_global_v_pos, cursor_global_h_pos

    def scroll_to_global_pos(self, global_v_pos, global_h_pos):
        """
        Change visible scrollbars value so that vpos/hpos is visible
        without changing the cursor position
        """
        model = self.model()
        widget = self.parent().parent()
        assert isinstance(widget, ArrayEditorWidget)
        visible_cols = widget.visible_cols()
        visible_rows = widget.visible_rows()

        hidden_v_offset = self.verticalScrollBar().value()
        hidden_h_offset = self.horizontalScrollBar().value()
        total_v_offset = model.v_offset + hidden_v_offset
        total_h_offset = model.h_offset + hidden_h_offset

        if global_v_pos < total_v_offset:
            new_total_v_offset = global_v_pos
        # TODO: document where those +2 come from
        elif global_v_pos > total_v_offset + visible_rows - 2:
            new_total_v_offset = global_v_pos - visible_rows + 2
        else:
            # do not change offset
            new_total_v_offset = total_v_offset

        if global_h_pos < total_h_offset:
            new_total_h_offset = global_h_pos
        elif global_h_pos > total_h_offset + visible_cols - 2:
            new_total_h_offset = global_h_pos - visible_cols + 2
        else:
            # do not change offset
            new_total_h_offset = total_h_offset

        # change visible scrollbars value
        widget.vscrollbar.setValue(new_total_v_offset)
        widget.hscrollbar.setValue(new_total_h_offset)

    def autofit_columns(self):
        """Resize cells to contents"""
        # print(f"{self.__class__.__name__}.autofit_columns()")
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        # for column in range(self.model_axes.columnCount()):
        #     self.resize_axes_column_to_contents(column)

        self.resizeColumnsToContents()
        # If the resized columns would make the whole view smaller or larger,
        # the view size itself (not its columns) is changed. This allows,
        # for example, other views (e.g. hlabels) to be moved accordingly.
        self.updateGeometry()
        QApplication.restoreOverrideCursor()

    def updateGeometry(self):
        # vpos = "TOP" if self.vpos == TOP else "BOTTOM"
        # hpos = "LEFT" if self.hpos == LEFT else "RIGHT"
        # print(f"{self.__class__.__name__}.updateGeometry() ({vpos=}, {hpos=})")
        # Set total height (for the whole view, not a particular row)
        if self.vpos == TOP:
            total_height = self.horizontalHeader().height() + \
                             sum(self.rowHeight(r) for r in range(self.model().rowCount()))
            # print(f"    TOP => {total_height=}")
            self.setFixedHeight(total_height)
        # Set total width (for the whole view, not a particular column)
        if self.hpos == LEFT:
            total_width = self.verticalHeader().width() + \
                            sum(self.columnWidth(c) for c in range(self.model().columnCount()))
            # print(f"    LEFT => {total_width=}")
            self.setFixedWidth(total_width)
        # update geometry
        super().updateGeometry()


class AxesView(AbstractView):
    """"Axes view class"""

    allSelected = Signal()

    def __init__(self, parent, model):
        # check model
        if not isinstance(model, AxesArrayModel):
            raise TypeError(f"Expected model of type {AxesArrayModel.__name__}. "
                            f"Received {type(model).__name__} instead")
        AbstractView.__init__(self, parent, model, LEFT, TOP)

        # FIXME: only have this if the adapter supports any extra action on axes
        # self.clicked.connect(self.on_clicked)

    def on_clicked(self, index: QModelIndex):
        row_idx = index.row()
        column_idx = index.column()

        # FIXME: column_idx works fine for the unfiltered/initial array but on an already filtered
        #        array it breaks because column_idx is the idx of the *filtered* array which can
        #        contain less axes while change_filter (via create_filter_menu) want the index
        #        of the *unfiltered* array
        try:
            adapter = self.model().adapter
            filtrable = adapter.can_filter_axis(column_idx)
            sortable = adapter.can_sort_axis_labels(column_idx)
            if sortable:
                sort_direction = adapter.axis_sort_direction(column_idx)
            else:
                sort_direction = 'unsorted'
            filter_labels = adapter.get_filter_options(column_idx)
        except IndexError:
            filtrable = False
            filter_labels = []
            sortable = False
            sort_direction = 'unsorted'
        if filtrable or sortable:
            menu = self.create_filter_menu(column_idx,
                                           filtrable,
                                           filter_labels,
                                           sortable,
                                           sort_direction)
            x = (self.columnViewportPosition(column_idx) +
                 self.verticalHeader().width())
            y = (self.rowViewportPosition(row_idx) + self.rowHeight(row_idx) +
                 self.horizontalHeader().height())
            menu.exec_(self.mapToGlobal(QPoint(x, y)))

    def create_filter_menu(self,
                           axis_idx,
                           filtrable,
                           filter_labels,
                           sortable=False,
                           sort_direction='unsorted'):
        def filter_changed(checked_items):
            # print("filter_changed", axis_idx, checked_items)
            array_widget = self.parent().parent()
            array_widget.filter_bar.change_filter(axis_idx, checked_items)

        def sort_changed(ascending):
            array_widget = self.parent().parent()
            array_widget.sort_axis_labels(axis_idx, ascending)

        menu = CombinedSortFilterMenu(self,
                                      filtrable=filtrable,
                                      sortable=sortable,
                                      sort_direction=sort_direction)
        if filtrable:
            menu.addItems([str(label) for label in filter_labels])
            menu.checked_items_changed.connect(filter_changed)
        if sortable:
            menu.sort_signal.connect(sort_changed)
        return menu

    # override viewOptions so that cell decorations (ie axes names arrows) are
    # drawn to the right of cells instead of to the left
    def viewOptions(self):
        option = QTableView.viewOptions(self)
        option.decorationPosition = QStyleOptionViewItem.Right
        return option

    def selectAll(self):
        self.allSelected.emit()


class LabelsView(AbstractView):
    """"Labels view class"""

    def __init__(self, parent, model, hpos, vpos):
        # check model
        if not isinstance(model, LabelsArrayModel):
            raise TypeError(f"Expected model of type {LabelsArrayModel.__name__}. "
                            f"Received {type(model).__name__} instead")
        AbstractView.__init__(self, parent, model, hpos, vpos)

        # FIXME: only have this if the adapter supports any extra action on axes
        if self.vpos == TOP:
            self.clicked.connect(self.on_clicked)

    def on_clicked(self, index: QModelIndex):
        if not index.isValid():
            return

        row_idx = index.row()
        local_col_idx = index.column()
        model: LabelsArrayModel = self.model()
        global_col_idx = model.h_offset + local_col_idx

        assert self.vpos == TOP

        # FIXME: global_col_idx works fine for the unfiltered/initial array but on
        #        an already filtered array it breaks because global_col_idx is the
        #        idx of the *filtered* array which can contain less axes while
        #        change_filter (via create_filter_menu) want the index of the
        #        *unfiltered* array
        adapter = model.adapter
        filtrable = adapter.can_filter_hlabel(1, global_col_idx)
        sortable = adapter.can_sort_hlabel(row_idx, global_col_idx)
        if sortable:
            sort_direction = adapter.hlabel_sort_direction(row_idx, global_col_idx)
            def sort_changed(ascending):
                # TODO: the chain for this is kinda convoluted:
                # local signal handler
                # -> ArrayWidget method
                # -> adapter method+model reset
                array_widget = self.parent().parent()
                array_widget.sort_hlabel(row_idx, global_col_idx, ascending)
        else:
            sort_direction = 'unsorted'
            sort_changed = None

        if filtrable:
            filter_labels = adapter.get_filter_options(global_col_idx)
            if len(filter_labels) == MAX_FILTER_OPTIONS:
                filter_labels = filter_labels.tolist()
                filter_labels[-1] = MORE_OPTIONS_NOT_SHOWN
            filter_indices = adapter.get_current_filter_indices(global_col_idx)
            def filter_changed(checked_items):
                # TODO: the chain for this is kinda convoluted:
                # local signal handler (this function)
                # -> ArrayWidget method
                # -> adapter method+model reset
                array_widget = self.parent().parent()
                assert isinstance(array_widget, ArrayEditorWidget)
                array_widget.filter_bar.change_filter(global_col_idx, checked_items)
        else:
            filter_labels = []
            filter_changed = None
            filter_indices = None

        if filtrable or sortable:
            # because of the local vs global idx, we cannot cache/reuse the
            # filter menu widget (we would need to remove the items and readd
            # the correct ones) so it is easier to just recreate the whole
            # widget. We need to take the already ticked indices into account
            # though.
            menu = self.create_filter_menu(global_col_idx,
                                           filter_labels,
                                           filter_indices,
                                           filter_changed,
                                           sort_changed,
                                           sort_direction)
            x = (self.columnViewportPosition(local_col_idx) +
                 self.verticalHeader().width())
            y = (self.rowViewportPosition(row_idx) + self.rowHeight(row_idx) +
                 self.horizontalHeader().height())
            menu.exec_(self.mapToGlobal(QPoint(x, y)))

    def create_filter_menu(self,
                           filter_idx,
                           filter_labels,
                           filter_indices,
                           filter_changed,
                           sort_changed,
                           sort_direction):
        filtrable = filter_changed is not None
        sortable = sort_changed is not None
        menu = CombinedSortFilterMenu(self,
                                      filtrable=filtrable,
                                      sortable=sortable,
                                      sort_direction=sort_direction)
        if filtrable:
            menu.addItems([str(label) for label in filter_labels],
                          filter_indices)
            # disable last item if there are too many options
            if len(filter_labels) == MAX_FILTER_OPTIONS:
                # this is correct (MAX - 1 to get the last item, + 1 because
                # of the "Select all" item at the beginning)
                last_item = menu._model[MAX_FILTER_OPTIONS - 1 + 1]
                last_item.setFlags(QtCore.Qt.NoItemFlags)

            menu.checked_items_changed.connect(filter_changed)
        if sortable:
            menu.sort_signal.connect(sort_changed)
        return menu

    # override viewOptions so that cell decorations (ie axes names arrows) are
    # drawn to the right of cells instead of to the left
    def viewOptions(self):
        option = QTableView.viewOptions(self)
        option.decorationPosition = QStyleOptionViewItem.Right
        return option


class ArrayDelegate(QItemDelegate):
    """Array Editor Item Delegate"""
    def __init__(self, parent=None, font=None, minvalue=None, maxvalue=None):
        # parent is the DataView instance
        QItemDelegate.__init__(self, parent)
        if font is None:
            font = get_default_font()
        self.font = font
        self.minvalue = minvalue
        self.maxvalue = maxvalue

        # keep track of whether there is already at least one editor already open (to properly
        # open a new editor when pressing Enter in DataView only if one is not already open)

        # We must keep a count instead of keeping a reference to the "current" one, because when switching
        # from one cell to the next, the new editor is created before the old one is destroyed, which means
        # it would be set to None when the old one is destroyed, instead of to the new current editor.
        self.editor_count = 0

    def createEditor(self, parent, option, index):
        """Create editor widget"""
        model = index.model()
        # TODO: dtype should be asked per cell. Only the adapter knows whether the dtype is per cell
        #  (e.g. list), per column (e.g. Dataframe) or homogenous for the whole table (e.g. la.Array)
        # dtype = model.adapter.get_dtype(hpos, vpos)

        dtype = model.adapter.dtype
        value = model.get_value(index)
        # this will return a string !
        # value = model.data(index, Qt.DisplayRole)
        if dtype.name == "bool":
            # directly toggle value and do not actually create an editor
            model.setData(index, not value)
            return None
        elif value is not np.ma.masked:
            # Not using a QSpinBox for integer inputs because I could not find
            # a way to prevent the spinbox/editor from closing if the value is
            # invalid. Using the builtin minimum/maximum of the spinbox works
            # but that provides no message so it is less clear.
            editor = QLineEdit(parent)
            if is_number_dtype(dtype):
                # FIXME: get minvalue & maxvalue from somewhere... the adapter?
                #        or the model? another specific adapter for minvalue,
                #        one for maxvalue, one for bg_value, etc.?
                minvalue, maxvalue = self.minvalue, self.maxvalue
                validator = QDoubleValidator(editor) if is_float_dtype(dtype) else QIntValidator(editor)
                if minvalue is not None:
                    validator.setBottom(minvalue)
                if maxvalue is not None:
                    validator.setTop(maxvalue)
                editor.setValidator(validator)

                if minvalue is not None and maxvalue is not None:
                    msg = f"value must be between {minvalue} and {maxvalue}"
                elif minvalue is not None:
                    msg = f"value must be >= {minvalue}"
                elif maxvalue is not None:
                    msg = f"value must be <= {maxvalue}"
                else:
                    msg = None

                if msg is not None:
                    def on_editor_text_edited():
                        if not editor.hasAcceptableInput():
                            QToolTip.showText(editor.mapToGlobal(QPoint()), msg)
                        else:
                            QToolTip.hideText()

                    editor.textEdited.connect(on_editor_text_edited)

            editor.setFont(self.font)
            editor.setAlignment(Qt.AlignRight)
            editor.destroyed.connect(self.on_editor_destroyed)
            self.editor_count += 1
            return editor

    def on_editor_destroyed(self):
        self.editor_count -= 1
        assert self.editor_count >= 0

    def setEditorData(self, editor, index):
        """Set editor widget's data"""
        text = index.model().data(index, Qt.DisplayRole)
        editor.setText(text)

    def setModelData(self, editor, model, index):
        parent = self.parent()
        assert isinstance(parent, DataView)
        # We store and recover scrollbar positions because the
        # model_data.reset() we do in EditObjectCommand, set the hidden
        # scrollbars to 0.
        hscrollbar = parent.horizontalScrollBar()
        vscrollbar = parent.verticalScrollBar()
        h_pos_before = hscrollbar.value()
        v_pos_before = vscrollbar.value()

        # This is the only thing we should be doing
        model.setData(index, editor.text())

        # recover original scrollbar positions
        hscrollbar.setValue(h_pos_before)
        vscrollbar.setValue(v_pos_before)


class DataView(AbstractView):
    """Data array view class"""

    signal_copy = Signal()
    signal_excel = Signal()
    signal_paste = Signal()
    signal_plot = Signal()

    def __init__(self, parent, model):
        # check model
        if not isinstance(model, DataArrayModel):
            raise TypeError(f"Expected model of type {DataArrayModel.__name__}. "
                            f"Received {type(model).__name__} instead")
        AbstractView.__init__(self, parent, model, RIGHT, BOTTOM)

        # adapter = model.adapter
        # available_actions = adapter.get_available_actions()
        self.context_menu = self.setup_context_menu()

        delegate = ArrayDelegate(self)
        self.setItemDelegate(delegate)
        self.doubleClicked.connect(self.activate_cell)

    def selectRow(self, buffer_v_pos: int):
        assert isinstance(buffer_v_pos, int)
        super().selectRow(buffer_v_pos)
        model: DataArrayModel = self.model()
        total_v_size, total_h_size = model.adapter.shape2d()
        global_v_pos = model.v_offset + buffer_v_pos
        self.first_selection_corner = (global_v_pos, 0)
        self.second_selection_corner = (global_v_pos, total_h_size - 1)

    def selectNewRow(self, buffer_v_pos: int):
        assert isinstance(buffer_v_pos, int)
        # if not MultiSelection mode activated, selectRow will unselect previously
        # selected rows (unless SHIFT or CTRL key is pressed)

        # this produces a selection with multiple QItemSelectionRange.
        # We could merge them here, but it is easier to handle in selection_bounds
        self.setSelectionMode(QTableView.MultiSelection)
        # do not call self.selectRow to avoid updating first_selection_corner
        super().selectRow(buffer_v_pos)
        self.setSelectionMode(QTableView.ContiguousSelection)

        model = self.model()
        total_v_size, total_h_size = model.adapter.shape2d()
        global_v_pos = model.v_offset + buffer_v_pos
        self.second_selection_corner = (global_v_pos, total_h_size - 1)

    def selectColumn(self, buffer_h_pos: int):
        assert isinstance(buffer_h_pos, int)
        super().selectColumn(buffer_h_pos)
        model = self.model()
        total_v_size, total_h_size = model.adapter.shape2d()
        global_h_pos = model.h_offset + buffer_h_pos
        self.first_selection_corner = (0, global_h_pos)
        self.second_selection_corner = (total_v_size - 1, global_h_pos)

    def selectNewColumn(self, buffer_h_pos: int):
        assert isinstance(buffer_h_pos, int)

        # if not MultiSelection mode activated, selectColumn will unselect previously
        # selected columns (unless SHIFT or CTRL key is pressed)
        # this produces a selection with multiple QItemSelectionRange. We could merge them here, but it is
        # easier to handle in selection_bounds
        self.setSelectionMode(QTableView.MultiSelection)
        # do not call self.selectColumn to avoid updating first_selection_corner
        super().selectColumn(buffer_h_pos)
        self.setSelectionMode(QTableView.ContiguousSelection)

        model = self.model()
        total_v_size, total_h_size = model.adapter.shape2d()
        global_h_pos = model.h_offset + buffer_h_pos
        self.second_selection_corner = (total_v_size - 1, global_h_pos)

    def selectAll(self):
        super().selectAll()
        total_v_size, total_h_size = self.model().adapter.shape2d()
        self.first_selection_corner = (0, 0)
        self.second_selection_corner = (total_v_size - 1, total_h_size - 1)

    def setup_context_menu(self):
        """Setup context menu"""
        actions_def = [
            (_('Copy'), keybinding('Copy'), 'edit-copy',
             lambda: self.signal_copy.emit()),
            (_('Copy to Excel'), "Ctrl+E", None,
             lambda: self.signal_excel.emit()),
            (_('Plot'), keybinding('Print'), None,
             lambda: self.signal_plot.emit()),
            (_('Paste'), keybinding('Paste'), 'edit-paste',
             lambda: self.signal_paste.emit()),
        ]
        actions = [
            create_action(self, label, shortcut=shortcut, icon=icon,
                          triggered=function)
            for label, shortcut, icon, function in actions_def
        ]
        menu = QMenu(self)
        menu.addActions(actions)

        # TODO: For some reason, when I wrote the context_menu code, the
        #       shortcuts from the actions in the context menu only worked
        #       if the widget had focus, EVEN when using
        #       action.setShortcutContext(Qt.ApplicationShortcut)
        #       (or Qt.WindowShortcut) so I had to redefine them here.
        #       I should revisit this code to see if that is still the case
        #       and even if so, I should do this in a cleaner way (probably by
        #       reusing the actions_def list above)
        shortcuts = [
            (keybinding('Copy'), self.parent().copy),
            (QKeySequence("Ctrl+E"), self.parent().to_excel),
            (keybinding('Paste'), self.parent().paste),
            (keybinding('Print'), self.parent().plot)
        ]
        for key_seq, target in shortcuts:
            shortcut = QShortcut(key_seq, self)
            shortcut.activated.connect(target)
        return menu

    def contextMenuEvent(self, event):
        """Reimplement Qt method"""
        self.context_menu.popup(event.globalPos())
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Reimplement Qt method"""
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            cursor_global_pos = self.get_cursor_global_pos()
            if cursor_global_pos is not None:
                self.first_selection_corner = cursor_global_pos

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Reimplement Qt method"""
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            cursor_global_pos = self.get_cursor_global_pos()
            if cursor_global_pos is not None:
                if self.first_selection_corner is not None:
                    # this is the normal case where we just finished a selection
                    self.second_selection_corner = cursor_global_pos
                else:
                    # this can happen when the array_widget is reset between
                    # a mouse button press and its release, e.g. when
                    # double-clicking in the explorer to open a dataset but
                    # keeping the button pressed during the second click,
                    # moving the mouse a bit to select the cell, then releasing
                    # the mouse button
                    self.first_selection_corner = None

    def keyPressEvent(self, event):
        """Reimplement Qt method"""

        # allow to start editing cells by pressing Enter
        if event.key() == Qt.Key_Return:
            index = self.currentIndex()
            # TODO: we should check whether the object is readonly
            #       before trying to activate. If an object is both
            #       editable and activatable, it will be a problem
            if not self.activate_cell(index):
                try:
                    # qt6
                    delegate = self.itemDelegateForIndex(index)
                except AttributeError:
                    # qt5
                    delegate = self.itemDelegate(index)
                if delegate.editor_count == 0:
                    self.edit(index)
        else:
            AbstractView.keyPressEvent(self, event)

    def selection_bounds(self):
        """
        Returns
        -------
        selection bounds (row_min, row_max, col_min, col_max -- end bounds are *exclusive*)
            If selection is empty, returns all data.
        """
        model = self.model()
        # We do not check/use the local "Qt" selection model, which can even
        # be empty even when something is selected if the view was scrolled
        # enough (via v/h_offset) that it went out of the buffer area
        if self.first_selection_corner is None:
            assert self.second_selection_corner is None
            total_rows, total_cols = model.adapter.shape2d()
            return 0, total_rows, 0, total_cols

        assert self.first_selection_corner is not None
        assert self.second_selection_corner is not None
        selection_v_pos1, selection_h_pos1 = self.first_selection_corner
        selection_v_pos2, selection_h_pos2 = self.second_selection_corner

        row_min = min(selection_v_pos1, selection_v_pos2)
        row_max = max(selection_v_pos1, selection_v_pos2)
        col_min = min(selection_h_pos1, selection_h_pos2)
        col_max = max(selection_h_pos1, selection_h_pos2)

        return row_min, row_max + 1, col_min, col_max + 1

    def activate_cell(self, index: QModelIndex):
        model = self.model()
        global_v_pos = model.v_offset + index.row()
        global_h_pos = model.h_offset + index.column()
        new_data = model.adapter.cell_activated(global_v_pos, global_h_pos)
        if new_data is not None:
            # the adapter wants us to open a sub-element
            array_widget = self.parent().parent()
            assert isinstance(array_widget, ArrayEditorWidget)
            adapter_creator = get_adapter_creator(new_data)
            assert adapter_creator is not None
            if isinstance(adapter_creator, str):
                QMessageBox.information(self, "Cannot display object",
                                        adapter_creator)
                return True

            from larray_editor.editor import AbstractEditorWindow, MappingEditorWindow
            widget = self
            while (widget is not None and
                   not isinstance(widget, AbstractEditorWindow) and
                   callable(widget.parent)):
                widget = widget.parent()
            if isinstance(widget, MappingEditorWindow):
                kernel = widget.ipython_kernel
                if kernel is not None:
                    # make the current object available in the console
                    kernel.shell.push({
                        '__current__': new_data
                    })
            if not (isinstance(new_data, Path) and new_data.is_dir()):
                # TODO: we should add an operand on the future quickbar instead
                array_widget.back_button_bar.add_back(array_widget.data,
                                                      array_widget.data_adapter)
            # TODO: we should open a new window instead (see above)
            array_widget.set_data(new_data)
            return True
        return False


class ScrollBar(QScrollBar):
    """
    A specialised scrollbar.
    """
    def __init__(self, parent, orientation, data_model, widget):
        super().__init__(orientation, parent)
        assert isinstance(data_model, DataArrayModel)
        assert isinstance(widget, ArrayEditorWidget)

        self.model = data_model
        self.widget = widget

        # We need to update_range when the *total* number of rows/columns
        # change, not when the loaded rows change so connecting to the
        # rowsInserted and columnsInserted signals is useless here
        data_model.modelReset.connect(self.update_range)

    def update_range(self):
        adapter = self.model.adapter
        if adapter is None:
            return
        # TODO: for some adapters shape2d is not reliable (it is a best guess),
        #       we should make sure we handle that
        total_rows, total_cols = adapter.shape2d()
        view_data = self.widget.view_data

        if self.orientation() == Qt.Horizontal:
            buffer_ncols = self.model.ncols
            hidden_hscroll_max = view_data.horizontalScrollBar().maximum()
            max_value = total_cols - buffer_ncols + hidden_hscroll_max
            logger.debug(f"update_range horizontal {total_cols=} {buffer_ncols=} {hidden_hscroll_max=} => {max_value=}")
            if total_cols == 0 and max_value != 0:
                logger.warn(f"empty data but {max_value=}. We let it pass for "
                            f"now (set it to 0).")
                max_value = 0
        else:
            buffer_nrows = self.model.nrows
            hidden_vscroll_max = view_data.verticalScrollBar().maximum()
            max_value = total_rows - buffer_nrows + hidden_vscroll_max
            logger.debug(f"update_range vertical {total_rows=} {buffer_nrows=} {hidden_vscroll_max=} => {max_value=}")
            if total_rows == 0 and max_value != 0:
                logger.warn(f"empty data but {max_value=}. We let it pass for "
                            f"now (set it to 0).")
                max_value = 0
        assert max_value >= 0, "max_value should not be negative"
        value_before = self.value()
        min_before = self.minimum()
        max_before = self.maximum()
        self.setMinimum(0)
        self.setMaximum(max_value)
        logger.debug(f"   min: {min_before} -> 0 / "
                     f"max: {max_before} -> {max_value} / "
                     f"value: {value_before} -> {self.value()}")


available_gradients = [
    ('white', None),
    # Hue, Saturation, Value, Alpha-channel
    ('red-blue', LinearGradient([(0, [0.99, 0.7, 1.0, 0.6]), (1, [0.66, 0.7, 1.0, 0.6])])),
    ('blue-red', LinearGradient([(0, [0.66, 0.7, 1.0, 0.6]), (1, [0.99, 0.7, 1.0, 0.6])])),
    ('red-white-blue', LinearGradient([(0, [.99, .85, 1., .6]),
                                       (0.5 - 1e-16, [.99, .15, 1., .6]),
                                       (0.5, [1., 0., 1., 1.]),
                                       (0.5 + 1e-16, [.66, .15, 1., .6]),
                                       (1, [.66, .85, 1., .6])])),
    ('blue-white-red', LinearGradient([(0, [.66, .85, 1., .6]),
                                       (0.5 - 1e-16, [.66, .15, 1., .6]),
                                       (0.5, [1., 0., 1., 1.]),
                                       (0.5 + 1e-16, [.99, .15, 1., .6]),
                                       (1, [.99, .85, 1., .6])])),
]
gradient_map = dict(available_gradients)


# this whole class assumes the font is the same for all the data model cells
# XXX: move all this to AbstractArrayModel?
class FontMetrics:
    def __init__(self, data_model):
        self.data_model = data_model
        self._cached_font = self.model_font

    @property
    def model_font(self):
        return self.data_model.role_defaults[Qt.FontRole]

    def font_changed(self):
        model_font = self.model_font
        if model_font is self._cached_font:
            return False
        changed = model_font == self._cached_font
        # update cached font even if not changed so that the "is" check is enough next time
        self._cached_font = model_font
        return changed

    @cached_property(font_changed)
    def str_width(self):
        return QFontMetrics(self._cached_font).horizontalAdvance

    @cached_property(font_changed)
    def digit_width(self):
        str_width = self.str_width
        return max(str_width(str(i)) for i in range(10))

    @cached_property(font_changed)
    def sign_width(self):
        return max(self.str_width('+'), self.str_width('-'))

    @cached_property(font_changed)
    def dot_width(self):
        return self.str_width('.')

    @cached_property(font_changed)
    def exp_width(self):
        return self.str_width('e')

    def get_numbers_width(self, int_digits, frac_digits=0, need_sign=False, scientific=False):
        if scientific:
            int_digits = 1
        margin_width = 8  # empirically measured
        digit_width = self.digit_width
        width = margin_width + int_digits * digit_width
        if frac_digits > 0:
            width += self.dot_width + frac_digits * digit_width
        if need_sign:
            width += self.sign_width
        if scientific:
            width += self.exp_width + self.sign_width + 2 * self.digit_width
        return width


class ArrayEditorWidget(QWidget):
    dataChanged = Signal(list)
    # milliseconds between a scroll event and updating cell sizes
    UPDATE_SIZES_FROM_CONTENT_DELAY = 100

    def __init__(self, parent, data=None, readonly=False, attributes=None, bg_gradient='blue-red',
                 minvalue=None, maxvalue=None, digits=None):
        QWidget.__init__(self, parent)
        assert bg_gradient in gradient_map
        if data is not None and np.isscalar(data):
            readonly = True
        self.readonly = readonly

        # prepare internal views and models
        self.model_axes = AxesArrayModel(parent=self) #, readonly=readonly)
        self.view_axes = AxesView(parent=self, model=self.model_axes)

        self.model_hlabels = HLabelsArrayModel(parent=self) #, readonly=readonly)
        self.view_hlabels = LabelsView(parent=self, model=self.model_hlabels, hpos=RIGHT, vpos=TOP)

        self.model_vlabels = VLabelsArrayModel(parent=self) #, readonly=readonly)
        self.view_vlabels = LabelsView(parent=self, model=self.model_vlabels, hpos=LEFT, vpos=BOTTOM)

        self.model_data = DataArrayModel(parent=self) #, readonly=readonly, minvalue=minvalue, maxvalue=maxvalue)
        self.view_data = DataView(parent=self, model=self.model_data)

        self.font_metrics = FontMetrics(self.model_data)

        # in case data is None
        self.data_adapter = None

        # Create visible vertical and horizontal scrollbars
        # TODO: when models "total" shape change (this is NOT model.nrows/ncols), we should update the range of
        # vscrollbar/hscrollbar. this is already partially done in ScrollBar (it listens to modelReset signal) but
        # this is not enough
        self.vscrollbar = ScrollBar(self, Qt.Vertical, self.model_data, self)
        self.vscrollbar.valueChanged.connect(self.visible_vscroll_changed)
        self.hscrollbar = ScrollBar(self, Qt.Horizontal, self.model_data, self)
        self.hscrollbar.valueChanged.connect(self.visible_hscroll_changed)

        axes_h_header = self.view_axes.horizontalHeader()
        axes_v_header = self.view_axes.verticalHeader()
        hlabels_h_header = self.view_hlabels.horizontalHeader()
        vlabels_v_header = self.view_vlabels.verticalHeader()

        # Propagate section resizing (left -> right and top -> bottom)
        axes_h_header.sectionResized.connect(self.on_axes_column_resized)
        axes_v_header.sectionResized.connect(self.on_axes_row_resized)
        hlabels_h_header.sectionResized.connect(self.on_hlabels_column_resized)
        vlabels_v_header.sectionResized.connect(self.on_vlabels_row_resized)

        # only useful for debugging
        # data_h_header = self.view_data.horizontalHeader()
        # data_h_header.sectionResized.connect(self.on_data_column_resized)

        # Propagate auto-resizing requests
        axes_h_header.sectionHandleDoubleClicked.connect(self.resize_axes_column_to_contents)
        hlabels_h_header.sectionHandleDoubleClicked.connect(self.resize_hlabels_column_to_contents)
        axes_v_header.sectionHandleDoubleClicked.connect(self.resize_axes_row_to_contents)
        vlabels_v_header.sectionHandleDoubleClicked.connect(self.resize_vlabels_row_to_contents)

        # synchronize specific methods
        self.view_axes.allSelected.connect(self.view_data.selectAll)
        self.view_data.signal_copy.connect(self.copy)
        self.view_data.signal_excel.connect(self.to_excel)
        self.view_data.signal_paste.connect(self.paste)
        self.view_data.signal_plot.connect(self.plot)

        # propagate changes (add new items in the QUndoStack attribute of MappingEditor)
        self.model_data.newChanges.connect(self.data_changed)

        # Synchronize scrolling of the different hidden scrollbars
        # data <--> hlabels
        hidden_data_hscrollbar = self.view_data.horizontalScrollBar()
        hidden_hlabels_hscrollbar = self.view_hlabels.horizontalScrollBar()
        hidden_data_hscrollbar.valueChanged.connect(
            hidden_hlabels_hscrollbar.setValue
        )
        hidden_hlabels_hscrollbar.valueChanged.connect(
            hidden_data_hscrollbar.setValue
        )

        # data <--> vlabels
        hidden_data_vscrollbar = self.view_data.verticalScrollBar()
        hidden_vlabels_vscrollbar = self.view_vlabels.verticalScrollBar()
        hidden_data_vscrollbar.valueChanged.connect(
            hidden_vlabels_vscrollbar.setValue
        )
        hidden_vlabels_vscrollbar.valueChanged.connect(
            hidden_data_vscrollbar.setValue
        )

        # Propagate range updates from hidden scrollbars to visible scrollbars
        # The ranges are updated when we resize the window or some columns/rows
        # and some of them quit or re-enter the viewport.
        # We do NOT need to propagate the hidden scrollbar value changes
        # because we scroll by entire columns/rows, so resizing columns/rows
        # does not change the scrollbar values
        def hidden_hscroll_range_changed(min_value: int, max_value: int):
            self.hscrollbar.update_range()
        hidden_data_hscrollbar.rangeChanged.connect(hidden_hscroll_range_changed)
        def hidden_vscroll_range_changed(min_value: int, max_value: int):
            self.vscrollbar.update_range()
        hidden_data_vscrollbar.rangeChanged.connect(hidden_vscroll_range_changed)

        # Synchronize selecting columns(rows) via hor.(vert.) header of x(y)labels view
        hlabels_h_header.sectionPressed.connect(self.view_data.selectColumn)
        hlabels_h_header.sectionEntered.connect(self.view_data.selectNewColumn)
        vlabels_v_header.sectionPressed.connect(self.view_data.selectRow)
        vlabels_v_header.sectionEntered.connect(self.view_data.selectNewRow)

        # following lines are required to keep usual selection color
        # when selecting rows/columns via headers of label views.
        # Otherwise, selected rows/columns appear in grey.
        self.view_data.setStyleSheet("""QTableView {
            selection-background-color: palette(highlight);
            selection-color: white;
        }""")

        # set external borders
        array_frame = QFrame(self)
        array_frame.setFrameStyle(QFrame.StyledPanel)
        # remove borders of internal tables
        self.view_axes.setFrameStyle(QFrame.NoFrame)
        self.view_hlabels.setFrameStyle(QFrame.NoFrame)
        self.view_vlabels.setFrameStyle(QFrame.NoFrame)
        self.view_data.setFrameStyle(QFrame.NoFrame)
        # Set layout of table views:
        # [ axes  ][hlabels]|V|
        # [vlabels][ data  ]|s|
        # |  H. scrollbar  |
        array_layout = QGridLayout()
        array_layout.addWidget(self.view_axes, 0, 0)
        array_layout.addWidget(self.view_hlabels, 0, 1)
        array_layout.addWidget(self.view_vlabels, 1, 0)
        self.view_data.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        array_layout.addWidget(self.view_data, 1, 1)
        array_layout.addWidget(self.vscrollbar, 0, 2, 2, 1)
        array_layout.addWidget(self.hscrollbar, 2, 0, 1, 2)
        array_layout.setSpacing(0)
        array_layout.setContentsMargins(0, 0, 0, 0)
        array_frame.setLayout(array_layout)

        # Set filters and buttons layout
        self.back_button_bar = BackButtonBar(self)
        self.filter_bar = FilterBar(self)
        self.btn_layout = QHBoxLayout()
        self.btn_layout.setAlignment(Qt.AlignLeft)

        # sometimes also called "Fractional digits" or "scale"
        label = QLabel("Decimal Places")
        self.btn_layout.addWidget(label)
        # default range is 0-99
        spin = QSpinBox(self)
        spin.valueChanged.connect(self.frac_digits_changed)
        # spin.setRange(-1, 99)
        # this is used when the widget has its minimum value
        # spin.setSpecialValueText("auto")
        self.digits_spinbox = spin
        self.btn_layout.addWidget(spin)

        scientific = QCheckBox(_('Scientific'))
        scientific.stateChanged.connect(self.scientific_changed)
        self.scientific_checkbox = scientific
        self.btn_layout.addWidget(scientific)
        self.use_scientific = False

        gradient_chooser = QComboBox()
        gradient_chooser.setMaximumSize(120, 20)
        gradient_chooser.setIconSize(QSize(100, 20))

        pixmap = QPixmap(100, 15)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)

        # add white option
        # 1 (y) and 13 (height) instead of 0 and 15 to have a transparent border around/between the gradients
        painter.fillRect(0, 1, 100, 13, Qt.white)
        gradient_chooser.addItem(QIcon(pixmap), "white")

        # add other options
        for name, gradient in available_gradients[1:]:
            qgradient = gradient.as_qgradient()

            # * fill with white because gradient can be transparent and if we do not "start from white", it skews the
            #   colors.
            # * 1 (y) and 13 (height) instead of 0 and 15 to have a transparent border around/between the gradients
            painter.fillRect(0, 1, 100, 13, Qt.white)
            painter.fillRect(0, 1, 100, 13, qgradient)
            gradient_chooser.addItem(QIcon(pixmap), name, gradient)

        # without this, we can crash python :)
        del painter, pixmap
        # select default gradient
        # requires Qt5+
        # gradient_chooser.setCurrentText(bg_gradient)
        gradient_chooser.setCurrentIndex(gradient_chooser.findText(bg_gradient))
        gradient_chooser.currentIndexChanged.connect(self.gradient_changed)
        self.btn_layout.addWidget(gradient_chooser)
        self.gradient_chooser = gradient_chooser

        # Set widget layout
        layout = QVBoxLayout()
        layout.addWidget(self.back_button_bar)
        layout.addWidget(self.filter_bar)
        layout.addWidget(array_frame)
        layout.addLayout(self.btn_layout)
        # left, top, right, bottom
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # set gradient
        self.model_data.set_bg_gradient(gradient_map[bg_gradient])

        # TODO: store detected_column_widths too so that it does not vary so
        #       much on scroll. Viewing test_api_larray.py is a good test for
        #       this.
        self.user_defined_hlabels_column_widths = {}
        self.user_defined_axes_column_widths = {}
        self.user_defined_vlabels_row_heights = {}
        self.user_defined_axes_row_heights = {}
        self.detected_hlabels_column_widths = {}
        self.detected_axes_column_widths = {}
        # TODO: find some more efficient structure to store them. 99.9%
        #       of rows will use the default height
        self.detected_vlabels_row_heights = {}
        self.detected_axes_row_heights = {}
        self._updating_hlabels_column_widths = False
        self._updating_axes_column_widths = False
        self._updating_vlabels_row_heights = False
        self._updating_axes_row_heights = False

        update_timer = QTimer(self)
        update_timer.setSingleShot(True)
        update_timer.setInterval(self.UPDATE_SIZES_FROM_CONTENT_DELAY)
        update_timer.timeout.connect(self.update_cell_sizes_from_content)
        self.update_cell_sizes_timer = update_timer

        # set data
        if data is not None:
            self.set_data(data, attributes=attributes, frac_digits=digits)

    def visible_cols(self, include_partial=True):
        """number of visible columns *including* partially visible ones"""

        view_data = self.view_data
        hidden_h_offset = view_data.horizontalScrollBar().value()
        view_width = view_data.width()
        last_visible_col_idx = view_data.columnAt(view_width - 1)

        # +1 because last_visible_col_idx is a 0-based index
        # if last_visible_col_idx == -1 it means the visible area is larger than
        # the array
        num_cols = last_visible_col_idx + 1 if last_visible_col_idx != -1 else self.model_data.ncols
        # clsname = self.__class__.__name__
        # logger.debug(f"{clsname}.visible_cols({include_partial=})")
        # logger.debug(f"    {hidden_h_offset=} {view_width=} "
        #              f"{last_visible_col_idx=} => {num_cols=}")
        if not include_partial and last_visible_col_idx != -1:
            last_visible_col_width = view_data.columnWidth(last_visible_col_idx)
            # no - 1 necessary here
            next_to_last_col_idx = view_data.columnAt(view_width -
                                                      last_visible_col_width)
            has_partial = next_to_last_col_idx < last_visible_col_idx
            if has_partial:
                num_cols -= 1
        return num_cols - hidden_h_offset

    def visible_rows(self, include_partial=True):
        """number of visible rows *including* partially visible ones"""

        view_data = self.view_data
        hidden_v_offset = view_data.verticalScrollBar().value()
        view_height = view_data.height()
        last_visible_row_idx = view_data.rowAt(view_height - 1)

        # +1 because last_visible_row_idx is a 0-based index
        # if last_visible_row_idx == -1 it means the visible area is larger than
        # the array
        num_rows = last_visible_row_idx + 1 if last_visible_row_idx != -1 else self.model_data.nrows
        # clsname = self.__class__.__name__
        # logger.debug(f"{clsname}.visible_rows({include_partial=})")
        # logger.debug(f"    {hidden_v_offset=} {view_height=} "
        #              f"{last_visible_row_idx=} => {num_rows=}")
        if not include_partial and last_visible_row_idx != -1:
            last_visible_row_height = view_data.rowHeight(last_visible_row_idx)
            # no - 1 necessary here
            next_to_last_row_idx = view_data.rowAt(view_height -
                                                   last_visible_row_height)
            has_partial = next_to_last_row_idx < last_visible_row_idx
            if has_partial:
                num_rows -= 1
                # logger.debug(f"    {has_partial=} => {num_rows=}")
        visible_rows = num_rows - hidden_v_offset
        # logger.debug(f"    {hidden_v_offset=} => {visible_rows=}")
        return visible_rows

    # Update the local/Qt selection, if needed
    def _update_selection(self, new_h_offset, new_v_offset):
        view_data = self.view_data

        selection_model = view_data.selectionModel()
        assert isinstance(selection_model, QItemSelectionModel)
        local_selection = selection_model.selection()
        assert isinstance(local_selection, QItemSelection)
        # if there is a local selection, we always need to move it (and
        # sometimes shrink it); if there is a global selection and no local
        # selection we may need to create a local selection when the global
        # selection intersects with the viewport
        global_selection_set = view_data.first_selection_corner is not None
        if local_selection or global_selection_set:
            model_data = self.model_data
            row_min, row_max, col_min, col_max = view_data.selection_bounds()

            # we need to clip local coordinates in case the selection
            # corners are outside the viewport
            local_top = max(row_min - new_v_offset, 0)
            local_left = max(col_min - new_h_offset, 0)
            # -1 because selection_bounds are exclusive while Qt use
            # inclusive bounds
            local_bottom = min(row_max - 1 - new_v_offset, model_data.nrows - 1)
            local_right = min(col_max - 1 - new_h_offset, model_data.ncols - 1)
            local_selection = QItemSelection(
                model_data.index(local_top, local_left),
                model_data.index(local_bottom, local_right)
            )
            selection_model.select(local_selection,
                                   QItemSelectionModel.ClearAndSelect)

    def visible_vscroll_changed(self, value):
        # 'value' will be the first visible row
        assert value >= 0, f"value must be >= 0 but is {value!r}"
        model_data = self.model_data
        hidden_vscroll = self.view_data.verticalScrollBar()
        # hidden_vscroll_max is the margin we got before we must move the buffer
        hidden_vscroll_max = hidden_vscroll.maximum()
        v_offset = model_data.v_offset
        extra_move = hidden_vscroll_max // 2
        logger.debug(f"visible vscroll changed({value=}, {v_offset=}, "
                     f"hidden_max={hidden_vscroll_max}, {extra_move=})")

        # buffer is beyond what is asked to display, we need to move it back
        if value < v_offset:
            # we could simply set it to value but we want to move more to avoid
            # fetching data for each row
            new_v_offset = max(value - extra_move, 0)
            msg = "    value < v_offset (min)"

        # we don't need to move the buffer (we can absorb the scroll change
        # entirely with the hidden scroll)
        elif value <= v_offset + hidden_vscroll_max:
            new_v_offset = v_offset
            msg = "    min <= value <= max => change hidden only"

        # buffer is before what is asked to display, we need to move it further
        #           <-visible_rows->
        #        <------nrows---------->
        # |      |------buffer---------|    |       |          |
        # ^      ^                          ^       ^          ^
        # 0      v_offset                   value   max_value  total_rows
        else:
            # we could simply set it to "value - hidden_vscroll_max" to move as
            # little as possible (this would place the visible rows at the end
            # of the buffer) but we want to move more to avoid fetching data
            # each time we move a single row
            new_v_offset = value - hidden_vscroll_max + extra_move
            # make sure we always have an entire buffer
            total_rows, total_cols = self.data_adapter.shape2d()
            new_v_offset = min(new_v_offset, total_rows - model_data.nrows)
            msg = "    value > v_offset + invis (max)"

        assert new_v_offset >= 0
        assert new_v_offset <= value <= new_v_offset + hidden_vscroll_max

        new_hidden_offset = value - new_v_offset
        logger.debug(f"{msg} => {new_hidden_offset=}, {new_v_offset=}")
        if new_v_offset != v_offset:
            model_data.set_v_offset(new_v_offset)
            self.model_vlabels.set_v_offset(new_v_offset)
            self._update_selection(model_data.h_offset, new_v_offset)
            self.update_cell_sizes_timer.start()

        hidden_vscroll.setValue(new_hidden_offset)

    def update_cell_sizes_from_content(self):
        logger.debug("ArrayEditorWidget.update_cell_sizes_from_content()")
        # TODO: having this in a timer alleviates the scrolling speed issue
        #       but we could also try to make this faster:
        #       * Would computing the sizeHint ourselves help?
        #       * For many in-memory (especially numerical) containers,
        #         it would be cheaper to compute that once for the
        #         whole array instead of after each scroll
        self._update_hlabels_column_widths_from_content()
        # we do not need to update axes cell size on scroll but vlabels
        # width can change on scroll (and they are linked to axes widths)
        self._update_vlabels_row_heights_from_content()
        self._update_axes_column_widths_from_content()
        self._update_axes_row_heights_from_content()

    def visible_hscroll_changed(self, value):
        # 'value' will be the first visible column
        assert value >= 0, f"value must be >= 0 but is {value!r}"
        model_data = self.model_data
        hidden_hscroll = self.view_data.horizontalScrollBar()
        # hidden_hscroll_max is the margin we got before we must move the buffer
        hidden_hscroll_max = hidden_hscroll.maximum()
        extra_move = hidden_hscroll_max // 2
        h_offset = model_data.h_offset
        logger.debug(f"visible hscroll changed ({value=}, {h_offset=}, "
                     f"hidden_max={hidden_hscroll_max}, {extra_move=})")

        # buffer is beyond what is asked to display, we need to move it back
        if value < h_offset:
            # we could simply set it to value but we want to move more to avoid
            # fetching data for each row
            new_h_offset = max(value - extra_move, 0)
            msg = "value < h_offset (min)"

        # we don't need to move the buffer (we can absorb the scroll change
        # entirely with the hidden scroll)
        elif value <= h_offset + hidden_hscroll_max:
            new_h_offset = h_offset
            msg = "min <= value <= max (hidden only)"

        # buffer is before what is asked to display, we need to move it further
        #           <-visible_cols->
        #        <------ncols---------->
        # |      |------buffer---------|    |       |          |
        # ^      ^                          ^       ^          ^
        # 0      h_offset                   value   max_value  total_cols
        else:
            # we could simply set it to "value - hidden_hscroll_max" to move as
            # little as possible (this would place the visible cols at the end
            # of the buffer) but we want to move more to avoid fetching data
            # each time we move a single col
            new_h_offset = value - hidden_hscroll_max + extra_move
            # make sure we always have an entire buffer
            total_rows, total_cols = self.data_adapter.shape2d()
            new_h_offset = min(new_h_offset, total_cols - model_data.ncols)
            msg = "value > h_offset + invis (max)"

        assert new_h_offset >= 0
        assert new_h_offset <= value <= new_h_offset + hidden_hscroll_max

        new_hidden_offset = value - new_h_offset
        logger.debug(f"{msg} => {new_hidden_offset=}, {new_h_offset=}")
        if new_h_offset != h_offset:
            model_data.set_h_offset(new_h_offset)
            self.model_hlabels.set_h_offset(new_h_offset)
            self._update_selection(new_h_offset, model_data.v_offset)
            self.update_cell_sizes_timer.start()
        hidden_hscroll.setValue(new_hidden_offset)

    def on_axes_column_resized(self, logical_index, old_size, new_size):
        # synchronize with linked view
        # equivalent (AFAICT) to:
        # view_vlabels.horizontalHeader().resizeSection(logical_index, new_size)
        self.view_vlabels.setColumnWidth(logical_index, new_size)
        if self._updating_axes_column_widths:
            return
        self.user_defined_axes_column_widths[logical_index] = new_size

    def on_axes_row_resized(self, logical_index, old_size, new_size):
        # synchronize with linked view
        self.view_hlabels.setRowHeight(logical_index, new_size)
        if self._updating_axes_row_heights:
            return
        self.user_defined_axes_row_heights[logical_index] = new_size

    def on_hlabels_column_resized(self, logical_index, old_size, new_size):
        # synchronize with linked view
        # logger.debug(f"on_hlabels_column_resized {logical_index=} {new_size=}")
        self.view_data.setColumnWidth(logical_index, new_size)
        if self._updating_hlabels_column_widths:
            return
        h_offset = self.model_data.h_offset
        self.user_defined_hlabels_column_widths[logical_index + h_offset] = new_size

    # def on_data_column_resized(self, logical_index, old_size, new_size):
        # log_caller()
        # logger.debug(f"on_data_column_resized {logical_index=} {new_size=}")

    def on_vlabels_row_resized(self, logical_index, old_size, new_size):
        # synchronize with linked view
        self.view_data.setRowHeight(logical_index, new_size)
        if self._updating_vlabels_row_heights:
            return
        v_offset = self.model_data.v_offset
        self.user_defined_vlabels_row_heights[logical_index + v_offset] = new_size

    def gradient_changed(self, index):
        gradient = self.gradient_chooser.itemData(index) if index > 0 else None
        self.model_data.set_bg_gradient(gradient)

    def data_changed(self, data_model_changes):
        global_changes = self.data_adapter.translate_changes(data_model_changes)
        self.dataChanged.emit(global_changes)

    def _set_models_adapter(self):
        self.model_axes.set_adapter(self.data_adapter)
        self.model_hlabels.set_adapter(self.data_adapter)
        self.model_vlabels.set_adapter(self.data_adapter)
        self.model_data.set_adapter(self.data_adapter)

    def set_data(self, data, attributes=None, frac_digits=None):
        # get new adapter instance + set data
        # TODO: add a mechanism that adapters can use to tell whether they support a
        #       particular *instance* of a data structure. This should probably be a
        #       class method.
        #       For example for memoryview, "structured"
        #       memoryview are not supported and get_adapter currently returns None
        data_adapter = get_adapter(data, attributes)
        if data_adapter is None:
            return
        self.data = data
        self.set_data_adapter(data_adapter, frac_digits)

    def close(self):
        logger.debug("ArrayEditorWidget.close()")
        if self.data_adapter is not None:
            self._close_adapter(self.data_adapter)
        self.back_button_bar.clear()
        super().close()

    @staticmethod
    def _close_adapter(adapter):
        clsname = type(adapter).__name__
        logger.debug(f"closing data adapter ({clsname})")
        adapter.close()

    def set_data_adapter(self, data_adapter: AbstractAdapter, frac_digits):
        old_adapter = self.data_adapter
        if old_adapter is not None:
            # We only need to close it if that adapter is not used in any
            # "back button"
            if not any(adapter is old_adapter
                       for adapter in self.back_button_bar._back_data_adapters):
                self._close_adapter(old_adapter)
        self.data_adapter = data_adapter

        # update models
        self._set_models_adapter()

        # reset widget to initial state
        self.reset_to_defaults()

        # update data format & autosize all cells
        # view_data and view_hlabels columns are resized automatically in
        # set_frac_digits_or_scientific, so using self.autofit_columns()
        # (which resizes columns of the 4 different views) is overkill but we
        # still need to resize view_axes and view_vlabels columns.
        self.set_frac_digits_or_scientific(frac_digits=frac_digits,
                                           scientific=None)
        self._update_axes_column_widths_from_content()
        self._update_axes_row_heights_from_content()
        self._update_vlabels_row_heights_from_content()

    def reset_to_defaults(self):
        logger.debug(f"{self.__class__.__name__}.reset_to_defaults()")

        # reset visible scrollbars
        self.vscrollbar.setValue(0)
        self.hscrollbar.setValue(0)

        # reset filters
        self.filter_bar.reset_to_defaults()

        # reset default sizes and clear selection
        self.view_axes.reset_to_defaults()
        self.view_vlabels.reset_to_defaults()
        self.view_hlabels.reset_to_defaults()
        self.view_data.reset_to_defaults()

        # clear user defined & detected column widths & row heights
        self.user_defined_axes_column_widths = {}
        self.user_defined_axes_row_heights = {}
        self.user_defined_hlabels_column_widths = {}
        self.user_defined_vlabels_row_heights = {}
        self.detected_hlabels_column_widths = {}
        self.detected_axes_column_widths = {}
        self.detected_vlabels_row_heights = {}
        self.detected_axes_row_heights = {}

    def set_frac_digits_or_scientific(self, frac_digits=None, scientific=None):
        """Set format.

        Parameters
        ----------
        frac_digits : int, optional
            Number of decimals to display. Defaults to None (autodetect).
        scientific : boolean, optional
            Whether or not to display values in scientific format.
            Defaults to None (autodetect).

        Currently, it is called from 3 places/cases:
         - set_data                 => frac_digits=None, scientific=None
         - user changes scientific  => frac_digits=None, scientific=bool
         - user changes frac_digits => frac_digits=int,  scientific=bool

        ON NEW DATA
            compute vmin/vmax on "start" buffer
                - can be per buffer, per column, or per row depending on adapter

            if api-provided ndigits is None and scientific is None (default)
                autodetect scientific & ndigits
                    - can be per buffer, per column, or per row
                autodetect column width
                    - can be per buffer, or per column
                autodetect row height
                    - can be per buffer or per row
            elif ndigits is a dict and scientific is a dict:
                autodetect column widths:
                    - should be per column, but unsure it is worth blocking
                      per buffer even though I do not think it makes sense
            elif ndigits is an int and scientific is a bool:
                autodetect column widths:
                    - can be per buffer or per column (depending on adapter)
                      if per column, ndigits is a modifier to autodetected
                      value
            elif ndigits is an int and scientific is None:
                autodetect scientific
                    - can be per buffer, per column, or per row
                autodetect column widths:
                    - can be per buffer or per column (depending on adapter)
                      if per column, ndigits is a modifier to autodetected
                      value
            elif ndigits is None and scientific is a bool:
                autodetect ndigits
                    - can be per buffer, per column, or per row
                autodetect column widths:
                    - can be per buffer or per column (depending on adapter)
                      if per column, GUI ndigits is a modifier to autodetected
                      value
        ON V_OFFSET CHANGE:
            update vmin/vmax (do not recompute on *just* the current buffer)
                - can be per buffer, per column or per row
            if LARRAY:
                IF still the "autodetected" ndigits,
                    re-detect ndigits given current window
                    (I don't think we should touch scientific in this case)
            elif dataframe:
                re-detect ndigits given current window
                    (I don't think we should touch scientific in this case)
                add/subtract ndigits offset
            update columns width if needed
        ON H_OFFSET change:
            update vmin/vmax
            update (invisible) columns width (do not change ndigits, scientific)
        ON SCIENTIFIC CHANGE:
            do NOT update column widths
            autodetect ndigits
                - can be per buffer, per column, per row or per cell
        ON NDIGITS CHANGE:
            update columns width
        ON COLWIDTH CHANGE:
            if dataframe or per-column array (*):
                re-detect ndigits only for the changed column
                (I don't think we should touch scientific in this case)
            elif homogeneous array:
                synchronize exact column width for all columns
                   doing this implicitly via ndigits will result in "unpleasant"
                   resizing I think
                re-detect scientific format for all columns


        one option: move format determination in the adapter or model

        when set_frac_digits_or_scientific from the UI:
            call corresponding method on adapter passing current column widths
            then force-fetch resulting data from the model to compute final
            column widths
        """
        logger.debug(f"ArrayEditorWidget.set_frac_digits_or_scientific("
                     f"{frac_digits=}, {scientific=})")
        assert frac_digits is None or isinstance(frac_digits, int)
        assert scientific is None or isinstance(scientific, bool)
        scientific_toggled = scientific is not None and scientific != self.use_scientific

        data_sample = self.data_adapter.get_sample()
        if not isinstance(data_sample, np.ndarray):
            # TODO: for non numpy homogeneous data types, this is suboptimal
            data_sample = np.asarray(data_sample, dtype=object)
        is_number_dtype = (isinstance(data_sample, np.ndarray) and
                           np.issubdtype(data_sample.dtype, np.number))
        cur_colwidth = self._get_current_min_col_width()

        if is_number_dtype and data_sample.size:
            # TODO: vmin/vmax should come from the adapter (were it is already
            #       computed and modified whenever the data changes)
            # TODO: some (all?) of this should be done in the adapter because
            #       it knows whether vmin/vmax should be per column or global
            #       and in the end if format and colwidth should be the same
            #       for the whole array or per col but I am still unsure of the
            #       boundary because font_metrics should not be used in the
            #       adapter.
            #       * The adapter also knows how expensive it is to compute some
            #         stuff and whether we can compute vmin/vmax on the full
            #         array or have to rely on sample + "rolling" vmin/vmax.
            #       * If vmin/vmax are arrays, we need to know which
            #         rows/columns (v_offset/h_offset) they correspond to.
            vmin, vmax = np.min(data_sample), np.max(data_sample)
            is_finite_data = np.isfinite(vmin) and np.isfinite(vmax)
            # logger.debug(f"    {data_sample=}")
            if is_finite_data:
                finite_sample = data_sample
                finite_vmin, finite_vmax = vmin, vmax
            else:
                isfinite = np.isfinite(data_sample)
                if isfinite.any():
                    finite_sample = data_sample[isfinite]
                    finite_vmin = np.min(finite_sample)
                    finite_vmax = np.max(finite_sample)
                else:
                    finite_sample = None
                    finite_vmin, finite_vmax = 0, 0
                    scientific = False
                    frac_digits = 0
            # logger.debug(f"    {finite_sample=}")
            logger.debug(f"    {finite_vmin=}, {finite_vmax=}")
            absmax = max(abs(finite_vmin), abs(finite_vmax))
            int_digits = num_int_digits(absmax)
            logger.debug(f"    {absmax=} {int_digits=}")
            has_negative = finite_vmin < 0

            font_metrics = self.font_metrics

            # choose whether or not to use scientific notation
            # ================================================
            if scientific is None:
                # TODO: use numpy ops so that it works for array inputs too

                # use scientific format if there are more integer digits than we can display or if we can display
                # more information that way (scientific format "uses" 4 digits, so we have a net win if we have
                # >= 4 zeros -- *including the integer one*)
                # TODO: only do so if we would actually display more information
                # 0.00001 can be displayed with 8 chars
                # 1e-05
                # would
                # logabsmax = np.where(absmax > 0, np.log10(absmax), 0)
                logabsmax = math.log10(absmax) if absmax else 0
                # minimum number of zeros before meaningful fractional part
                # frac_zeros = np.where(logabsmax < 0, np.ceil(-logabsmax) - 1, 0)
                frac_zeros = math.ceil(-logabsmax) - 1 if logabsmax < 0 else 0
                non_scientific_int_width = font_metrics.get_numbers_width(int_digits, need_sign=has_negative)
                # with the current default width and font size, this accepts up
                # to 8 digits for positive numbers (7 for negative)
                # TODO: change that to accept up to 12 digits for positive
                #       numbers (11 for negative) so that with the thousand
                #       separators we can display values up to 999 billions
                #       without using scientific notation
                # scientific = (non_scientific_int_width > cur_colwidth) | (frac_zeros >= 4)
                scientific = non_scientific_int_width > cur_colwidth or frac_zeros >= 4
                logger.debug(f"    {logabsmax=} {frac_zeros=} {non_scientific_int_width=}")
                logger.info(f"     -> detected scientific={scientific}")

            # determine best number of decimals to display
            # ============================================
            if frac_digits is None:
                int_part_width = font_metrics.get_numbers_width(int_digits, need_sign=has_negative,
                                                                scientific=scientific)
                # logger.debug(f"    {int_digits=} {has_negative=} {scientific=} => {int_part_width=}")
                # since we are computing the number of frac digits, we always need the dot
                avail_width_for_frac_part = max(cur_colwidth - int_part_width - font_metrics.dot_width, 0)
                # logger.debug(f"    {cur_colwidth=} {font_metrics.dot_width=} => {avail_width_for_frac_part=}")
                max_frac_digits = avail_width_for_frac_part // font_metrics.digit_width
                # logger.debug(f"    {font_metrics.digit_width=} => {max_frac_digits=}")
                frac_digits = data_frac_digits(finite_sample, max_frac_digits=max_frac_digits)
                # logger.info(f"     -> detected {frac_digits=}")

            format_letter = 'e' if scientific else 'f'
            fmt = '%%.%d%s' % (frac_digits, format_letter)
            data_colwidth = (
                font_metrics.get_numbers_width(int_digits,
                                               frac_digits,
                                               need_sign=has_negative,
                                               scientific=scientific))
            if not is_finite_data:
                # We have nans or infs, so we have to make sure we have enough
                # room to display "nan" or "inf"
                # ideally we should add a finite_sample argument to
                # get_numbers_width so that we take the actual "nan" and "inf"
                # strings width but I am unsure it is worth it.
                # Especially given this whole thing is almost useless (it only
                # serves to trigger the data_colwidth > cur_colwidth condition
                # so that the column widths are re-computed & updated)
                inf_nan_colwidth = (
                    font_metrics.get_numbers_width(3,
                                                   need_sign=vmin == -math.inf))
                data_colwidth = max(data_colwidth, inf_nan_colwidth)
        else:
            frac_digits = 0
            scientific = False
            fmt = '%s'
            data_colwidth = 0

        self.data_adapter.set_format(fmt)
        self.model_data._get_current_data()
        self.model_data.reset()

        self.use_scientific = scientific

        # avoid triggering frac_digits_changed which would cause a useless redraw
        self.digits_spinbox.blockSignals(True)
        self.digits_spinbox.setValue(frac_digits)
        self.digits_spinbox.setEnabled(is_number_dtype)
        self.digits_spinbox.blockSignals(False)

        # avoid triggering scientific_changed which would call this function a second time
        self.scientific_checkbox.blockSignals(True)
        self.scientific_checkbox.setChecked(scientific)
        self.scientific_checkbox.setEnabled(is_number_dtype)
        self.scientific_checkbox.blockSignals(False)

        frac_digits_changed = not scientific_toggled
        # frac digits changed => set new column width
        if frac_digits_changed or data_colwidth == 0 or data_colwidth > cur_colwidth:
            self._update_hlabels_column_widths_from_content()

    def _update_hlabels_column_widths_from_content(self):
        h_offset = self.model_data.h_offset
        hlabels_header = self.view_hlabels.horizontalHeader()

        # TODO: I wonder if we should only set the widths for the visible
        #       columns + some margin as the buffer could become relatively
        #       large if we let adapters decide their size
        user_def_col_width = self.user_defined_hlabels_column_widths

        # This prevents the auto column width changes below from updating
        # user_defined_column_widths (via the sectionResized signal).
        # This is ugly but I found no other way to avoid that because using
        # hlabels_header.blockSignals(True) breaks updating the column width
        # in response to ndigits changes (it updates the header widths but not
        # the hlabels cells widths).
        self._updating_hlabels_column_widths = True
        for local_col_idx in range(self.model_data.columnCount()):
            global_col_idx = h_offset + local_col_idx

            # We should NOT take the max of the user defined column width and
            # the computed column width because sometimes the user wants
            # a _smaller_ width than the auto-detected one
            if global_col_idx in user_def_col_width:
                hlabels_header.resizeSection(local_col_idx,
                                             user_def_col_width[global_col_idx])
            else:
                self.resize_hlabels_column_to_contents(local_col_idx,
                                                       MIN_COLUMN_WIDTH, MAX_COLUMN_WIDTH)
        self._updating_hlabels_column_widths = False

    def _update_vlabels_row_heights_from_content(self):
        vlabels_header = self.view_vlabels.verticalHeader()
        v_offset = self.model_data.v_offset
        self._updating_vlabels_row_heights = True
        user_def_row_heights = self.user_defined_vlabels_row_heights
        for local_row_idx in range(self.model_vlabels.rowCount()):
            global_row_idx = v_offset + local_row_idx
            if global_row_idx in user_def_row_heights:
                vlabels_header.resizeSection(local_row_idx,
                                             user_def_row_heights[global_row_idx])
            else:
                self.resize_vlabels_row_to_contents(local_row_idx)
        self._updating_vlabels_row_heights = False

    def _update_axes_column_widths_from_content(self):
        self._updating_axes_column_widths = True
        user_widths = self.user_defined_axes_column_widths
        for local_col_idx in range(self.model_axes.columnCount()):
            # Since there is no h_offset for axes, the column width never
            # actually changes unless the user explicitly changes it, so just
            # preventing the auto-sizing code from running is enough
            if local_col_idx not in user_widths:
                self.resize_axes_column_to_contents(local_col_idx)
        self._updating_axes_column_widths = False

    def _update_axes_row_heights_from_content(self):
        self._updating_axes_row_heights = True
        user_def_row_heights = self.user_defined_axes_row_heights
        for local_row_idx in range(self.model_axes.rowCount()):
            # Since there is no v_offset for axes, the row height never
            # actually changes unless the user explicitly changes it, so just
            # preventing the auto-sizing code from running is enough
            if local_row_idx not in user_def_row_heights:
                self.resize_axes_row_to_contents(local_row_idx)
        self._updating_axes_row_heights = False

    def _get_current_min_col_width(self):
        header = self.view_hlabels.horizontalHeader()
        if header.count():
            return min(header.sectionSize(i) for i in range(header.count()))
        else:
            return 0

    # must be connected to signal:
    # view_axes.horizontalHeader().sectionHandleDoubleClicked
    def resize_axes_column_to_contents(self, col_idx):
        # clsname = self.__class__.__name__
        # print(f"{clsname}.resize_axes_column_to_contents({col_idx})")
        # TODO:
        #  * maybe reimplement resizeColumnToContents(column) instead? Though
        #    the doc says it only resize visible columns, so that might not
        #    work
        #  * reimplementing sizeHintForColumn on AxesView would be cleaner
        #    but that would require making it know of the view_vlabels instance
        prev_width = self.detected_axes_column_widths.get(col_idx, 0)
        width = max(self.view_axes.sizeHintForColumn(col_idx),
                    self.view_vlabels.sizeHintForColumn(col_idx),
                    prev_width)
        # view_vlabels column width will be synchronized automatically
        self.view_axes.horizontalHeader().resizeSection(col_idx, width)

        # set that column's width back to "automatic width"
        if col_idx in self.user_defined_axes_column_widths:
            del self.user_defined_axes_column_widths[col_idx]
        if width > prev_width:
            self.detected_axes_column_widths[col_idx] = width

    # must be connected to signal:
    # view_hlabels.horizontalHeader().sectionHandleDoubleClicked
    def resize_hlabels_column_to_contents(self, local_col_idx,
                                          min_width=None, max_width=None):
        global_col_idx = self.model_data.h_offset + local_col_idx
        prev_width = self.detected_hlabels_column_widths.get(global_col_idx, 0)
        # logger.debug("ArrayEditorWidget.resize_hlabels_column_to_contents("
        #              f"{local_col_idx=}, {min_width=}, {max_width=})")
        width = max(self.view_hlabels.sizeHintForColumn(local_col_idx),
                    self.view_data.sizeHintForColumn(local_col_idx),
                    prev_width)
        # logger.debug(f"   {global_col_idx=} {prev_width=} => (before clip) "
        #              f"{width=} ")
        if min_width is not None:
            width = max(width, min_width)
        if max_width is not None:
            width = min(width, max_width)
        # logger.debug(f"   -> (after clip) {width=}")
        # view_data column width will be synchronized automatically
        self.view_hlabels.horizontalHeader().resizeSection(local_col_idx, width)

        # set that column's width back to "automatic width"
        if global_col_idx in self.user_defined_hlabels_column_widths:
            del self.user_defined_hlabels_column_widths[global_col_idx]
        if width > prev_width:
            # logger.debug(f"   -> width > prev_width (updating detected)")
            self.detected_hlabels_column_widths[global_col_idx] = width

    # must be connected to signal:
    # view_axes.verticalHeader().sectionHandleDoubleClicked
    def resize_axes_row_to_contents(self, row_idx):
        # clsname = self.__class__.__name__
        # print(f"{clsname}.resize_axes_row_to_contents({row})")
        prev_height = self.detected_axes_row_heights.get(row_idx, 0)
        height = max(self.view_axes.sizeHintForRow(row_idx),
                     self.view_hlabels.sizeHintForRow(row_idx),
                     prev_height)
        # view_hlabels row height will be synchronized automatically
        self.view_axes.verticalHeader().resizeSection(row_idx, height)
        # set that row's height back to "automatic height"
        if row_idx in self.user_defined_axes_row_heights:
            del self.user_defined_axes_row_heights[row_idx]
        if height > prev_height:
             self.detected_axes_row_heights[row_idx] = height

    # must be connected to signal:
    # view_vlabels.verticalHeader().sectionHandleDoubleClicked
    def resize_vlabels_row_to_contents(self, local_row_idx):
        # clsname = self.__class__.__name__
        # print(f"{clsname}.resize_vlabels_row_to_contents({row})")
        global_row_idx = self.model_data.v_offset + local_row_idx
        prev_height = self.detected_vlabels_row_heights.get(global_row_idx, 0)
        height = max(self.view_vlabels.sizeHintForRow(local_row_idx),
                     self.view_data.sizeHintForRow(local_row_idx),
                     prev_height)
        # view_data row height will be synchronized automatically
        self.view_vlabels.verticalHeader().resizeSection(local_row_idx, height)
        # set that row's height back to "automatic height"
        if global_row_idx in self.user_defined_vlabels_row_heights:
            del self.user_defined_vlabels_row_heights[global_row_idx]
        if height > prev_height:
             self.detected_vlabels_row_heights[global_row_idx] = height

    def scientific_changed(self, value):
        # auto-detect frac_digits
        self.set_frac_digits_or_scientific(frac_digits=None, scientific=bool(value))

    def frac_digits_changed(self, value):
        # TODO: I should probably drop the use_scientific field and just
        #       retrieve the checkbox value
        self.set_frac_digits_or_scientific(value, self.use_scientific)

    def sort_axis_labels(self, axis_idx, ascending):
        self.data_adapter.sort_axis_labels(axis_idx, ascending)
        self._set_models_adapter()

    def sort_hlabel(self, row_idx, col_idx, ascending):
        self.data_adapter.sort_hlabel(row_idx, col_idx, ascending)
        self._set_models_adapter()
        # since we will probably display different rows, they can have different
        # column widths
        self.update_cell_sizes_from_content()

    def copy(self):
        """Copy selection as text to clipboard"""
        text = self.data_adapter.to_string(*self.view_data.selection_bounds())
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    def to_excel(self):
        """Export selection in Excel"""
        try:
            self.data_adapter.to_excel(*self.view_data.selection_bounds())
        except ImportError:
            msg = "to_excel() is not available because xlwings is not installed"
            QMessageBox.critical(self, "Error", msg)

    def paste(self):
        # FIXME: this now returns coordinates in global space while the rest of
        #        this function assumes local/buffer space coordinates. But this
        #        whole "set_values" code should be revisited entirely anyway
        row_min, row_max, col_min, col_max = self.view_data.selection_bounds()
        clipboard = QApplication.clipboard()
        text = str(clipboard.text())
        list_data = [line.split('\t') for line in text.splitlines()]
        list_data = self.data_adapter.from_clipboard_data_to_model_data(list_data)
        new_data = np.array(list_data)
        if new_data.shape[0] > 1:
            row_max = row_min + new_data.shape[0]
        if new_data.shape[1] > 1:
            col_max = col_min + new_data.shape[1]

        # FIXME: the way to change data is extremely convoluted (and slightly wrong):
        #        * either Widget.paste or ArrayDelegate.setModelData (via model.setData)
        #        * calls model.set_values on the model
        #            that does not change any data but computes a
        #            {global_filtered_2D_coords: (old_value, new_value)}
        #            dict of changes
        #        * emits a newChanges signal and then a dataChanged (builtin)
        #          signal (which only works/makes sense because a signal .emit
        #          call only returns when all its slots have executed, hence the
        #          whole chain below has already been executed when that second
        #          signal is emitted).
        #        * the newChanges signal is caught by the widget, which
        #        * asks the adapter to transform the changes from 2d global (but
        #          potentially filtered) positional keys to ND global positional
        #          keys, then
        #        * re-emits a dataChanged signal with a list of those changes,
        #        * the editor catches that signal and
        #        * push those changes to the edit_undo_stack which actually
        #        * applies each change by using
        #           kernel.shell.run_cell(f"{self.target}.i[{key}] = {new_value}")
        #           OR
        #           self.target.i[key] = new_value
        #           and there,
        #        * editor.array_widget.model_data.reset() is called which
        #        * notifies qt the whole thing needs to be refreshed (including
        #          reprocessing the data via _process_data but does *NOT* fetch
        #          the actual new data!!!)
        #        and it actually only appears to work in the simple case of
        #        editing an unfiltered array because we are using array *views*
        #        all the way so when we edit the array, the "raw_data" in the
        #        model is updated directly too and _process_data is indeed
        #        enough.
        #        >
        #        Since we can *NOT* push a command on the edit_undo_stack
        #        without executing it, we should:
        #        * create widget.set_values method, call it from paste and the
        #          ArrayDelegate
        #          - ask the adapter to create an edit_undo_stack command (which
        #            will change the real data)
        #            * create a {NDkey: changes}
        #          - push command
        #          - call a new method on the model akin to reset() but which
        #            *fetches* the data in addition to processing it
        #          - we will probably need to emit/use signals in there but this
        #            can come later
        #        I am still undecided on whether the commands should actually
        #        update the live object or add changes to a "changes layer",
        #        which can later be all applied to the real objects. For
        #        in-memory objects, updating the objects directly seem better
        #        so that e.g. console commands stay consistent with what we see
        #        but for on-disk data, writing each change directly to disk
        #        seems inefficient and surprising. I suppose that decision
        #        should be done (and implemented) by the adapter but I have no
        #        idea how to convey the difference to users. It should be
        #        obvious but unobstrusive and users will need a way to trigger
        #        a "save". In any case, this can come later as we currently
        #        do not have any disk-backed adapter anywhere close to
        #        supporting editing values.
        #
        #        as a side note the (visible) Scrollbar is connected to the
        #        reset event and updates its range in that
        #        case which is useless
        result = self.model_data.set_values(row_min, col_min, row_max, col_max, new_data)
        if result is None:
            return

        # TODO: when pasting near bottom/right boundaries and size of
        # new_data exceeds destination size, we should either have an error
        # or clip new_data
        self.view_data.selectionModel().select(QItemSelection(*result), QItemSelectionModel.ClearAndSelect)

    def plot(self):
        from larray_editor.utils import show_figure
        from larray_editor.editor import AbstractEditorWindow, MappingEditorWindow
        try:
            figure = self.data_adapter.plot(*self.view_data.selection_bounds())
            widget = self
            while widget is not None and not isinstance(widget, AbstractEditorWindow) and callable(widget.parent):
                widget = widget.parent()
            title = widget.current_expr_text if isinstance(widget, MappingEditorWindow) else None
            show_figure(figure, title, parent=self)
        except ImportError:
            QMessageBox.critical(self, "Error", "plot() is not available because matplotlib is not installed")
