# -*- coding: utf-8 -*-
#
# Copyright © 2009-2012 Pierre Raybaut
# Copyright © 2015-2016 Gaëtan de Menten
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
# * drag & drop to reorder axes
#   http://zetcode.com/gui/pyqt4/dragdrop/
#   http://stackoverflow.com/questions/10264040/
#       how-to-drag-and-drop-into-a-qtablewidget-pyqt
#   http://stackoverflow.com/questions/3458542/multiple-drag-and-drop-in-pyqt4
#   http://ux.stackexchange.com/questions/34158/
#       how-to-make-it-obvious-that-you-can-drag-things-that-you-normally-cant
# * keep header columns & rows visible ("frozen")
#   http://doc.qt.io/qt-5/qtwidgets-itemviews-frozencolumn-example.html
# * document default icons situation (limitations)
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
# * view/edit DataFrames too
# * view/edit LArray over Pandas (ie sparse)
# * resubmit editor back for inclusion in Spyder
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
import logging

import numpy as np

from qtpy import QtCore
from qtpy.QtCore import Qt, QPoint, QItemSelection, QItemSelectionModel, Signal, QSize
from qtpy.QtGui import (QDoubleValidator, QIntValidator, QKeySequence, QFontMetrics, QCursor, QPixmap, QPainter, QIcon,
                        QResizeEvent, QWheelEvent, QMouseEvent)
from qtpy.QtWidgets import (QApplication, QTableView, QItemDelegate, QLineEdit, QCheckBox,
                            QMessageBox, QMenu, QLabel, QSpinBox, QWidget, QToolTip, QShortcut, QScrollBar,
                            QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy, QFrame, QComboBox)

from larray_editor.utils import (keybinding, create_action, clear_layout, get_default_font, is_number_dtype, is_float_dtype, _,
                                 ima, LinearGradient, logger, cached_property)
from larray_editor.arrayadapter import get_adapter
from larray_editor.arraymodel import (HLabelsArrayModel, VLabelsArrayModel, LabelsArrayModel,
                                      AxesArrayModel, DataArrayModel)
from larray_editor.combo import FilterComboBox


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
        # handling a second selection corner is necessary to implement the "select entire row/column" functionality
        # because in that case the second corner is *not* in the viewport
        self.second_selection_corner = None
        self.hidden_hscrollbar_oldvalue = 0
        self.hidden_vscrollbar_oldvalue = 0

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

        # to fetch more rows/columns when required
        self.horizontalScrollBar().valueChanged.connect(self.on_horizontal_scroll_changed)
        self.verticalScrollBar().valueChanged.connect(self.on_vertical_scroll_changed)

        # Hide scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # update geometry
        if not (hpos == RIGHT and vpos == BOTTOM):
            self.model().modelReset.connect(self.updateGeometry)
            self.horizontalHeader().sectionResized.connect(self.updateGeometry)
            self.verticalHeader().sectionResized.connect(self.updateGeometry)

    def selectionChanged(self, selected: QtCore.QItemSelection, deselected: QtCore.QItemSelection) -> None:
        super().selectionChanged(selected, deselected)
        print(f"selectionChanged: selected({display_selection(selected)}), deselected({display_selection(deselected)})")

    def set_default_size(self):
        # make the grid a bit more compact
        self.horizontalHeader().setDefaultSectionSize(64)
        self.verticalHeader().setDefaultSectionSize(20)
        if self.vpos == TOP:
            self.horizontalHeader().setFixedHeight(10)
        if self.hpos == LEFT:
            self.verticalHeader().setFixedWidth(10)

    def on_vertical_scroll_changed(self, value):
        parent = self.parent().parent()
        assert isinstance(parent, ArrayEditorWidget)
        vscrollbar = parent.vscrollbar
        # hscrollbar = parent.hscrollbar
        # vscrollbar = self.verticalScrollBar()
        # if more than one row selected
        #     fetchmorerows
        # else:
        #     move_offset
        print(f"hidden vscroll on {self.__class__.__name__} changed {value}")
        # vscrollbar.setValue(new_value)
        # model = self.model()
        # model.set_v_offset(new_offset, reset=False)
        #
        # if value == vscrollbar.minimum():
        #     diff = value - self.vscrollbar_oldvalue
        #     # assert diff < 0
        #     model = self.model()
        #     new_offset = model.v_offset + diff
        #     model.set_v_offset(new_offset, reset=False)
        #     vscrollbar.setValue(self.vscrollbar_oldvalue)
        # elif value == vscrollbar.maximum():
        #     diff = value - self.vscrollbar_oldvalue
        #     print(" == max => moving real scrollbar instead by", diff)
        #     if diff > 0:
        #         model = self.model()
        #         new_offset = model.v_offset + diff
        #         model.set_v_offset(new_offset, reset=False)
        #         vscrollbar.blockSignals(True)
        #         vscrollbar.setValue(self.vscrollbar_oldvalue)
        #         vscrollbar.blockSignals(False)
        #     # self.model().fetch_more_rows()
        # else:
        self.hidden_vscrollbar_oldvalue = value

    def on_horizontal_scroll_changed(self, value):
        parent = self.parent().parent()
        assert isinstance(parent, ArrayEditorWidget)
        hscrollbar = parent.hscrollbar
        # hscrollbar = self.horizontalScrollBar()
        # if more than one row selected
        #     fetchmorerows
        # else:
        #     move_offset
        print(f"hidden hscroll on {self.__class__.__name__} changed {value}")
        # hscrollbar.setValue(new_value)

        # print("hidden hscroll changed", value)
        # if value == self.horizontalScrollBar().maximum():
        #     diff = value - self.hscrollbar_oldvalue
        #     assert diff > 0
        #     model = self.model()
        #     new_offset = model.h_offset + diff
        #     model.set_h_offset(new_offset, reset=False)
        #     # self.model().fetch_more_columns()
        self.hidden_hscrollbar_oldvalue = value

    # We need to have this here (in AbstractView) and not only on DataView, so that we
    # catch them for vlabels too. For axes and hlabels, it is a bit of a weird
    # behavior since they are not affected themselves but that is really a nitpick
    # Also, overriding the general event() method for this does not work as it is
    # handled behind us (by the ScrollArea I assume) and we do not even see the event
    # unless we are at the buffer boundary.
    def wheelEvent(self, event: QWheelEvent):
        """Catch wheel events and send them to the corresponding visible scrollbar"""
        delta = event.angleDelta()
        print(f"wheelEvent on {self.__class__.__name__} ({delta})")
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
        print()
        print("navigate")
        print("========")
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

        keyseq = QKeySequence(event_modifiers | event_key)
        page_step = self.verticalScrollBar().pageStep()
        cursor_global_v_pos, cursor_global_h_pos = self.get_cursor_global_pos()
        print("old global cursor", cursor_global_v_pos, cursor_global_h_pos)

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
        print("new global cursor", cursor_new_global_v_pos, cursor_new_global_h_pos)

        self.scroll_to_global_pos(cursor_new_global_v_pos, cursor_new_global_h_pos)

        new_v_posinbuffer = cursor_new_global_v_pos - model.v_offset
        new_h_posinbuffer = cursor_new_global_h_pos - model.h_offset

        local_cursor_index = model.index(new_v_posinbuffer, new_h_posinbuffer)
        if shift:
            if self.first_selection_corner is None:
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
            local_bottom = min(row_max - model.v_offset, model.nrows)
            local_right = min(col_max - model.h_offset, model.ncols)
            selection = QItemSelection(model.index(local_top, local_left),
                                       model.index(local_bottom, local_right))
            selection_model.select(selection, QItemSelectionModel.ClearAndSelect)
        else:
            self.first_selection_corner = None
            self.second_selection_corner = None
            self.setCurrentIndex(local_cursor_index)

    def get_cursor_global_pos(self):
        model = self.model()
        current_index = self.currentIndex()
        v_posinbuffer = current_index.row()
        h_posinbuffer = current_index.column()
        cursor_global_v_pos = model.v_offset + v_posinbuffer
        cursor_global_h_pos = model.h_offset + h_posinbuffer
        return cursor_global_v_pos, cursor_global_h_pos

    def scroll_to_global_pos(self, global_v_pos, global_h_pos):
        """change visible scrollbars value so that vpos/hpos is visible"""
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
        elif global_v_pos > total_v_offset + visible_rows - 2:
            new_total_v_offset = global_v_pos - visible_rows + 2
        else:
            new_total_v_offset = total_v_offset

        if global_h_pos < total_h_offset:
            new_total_h_offset = global_h_pos
        elif global_h_pos > total_h_offset + visible_cols - 2:
            new_total_h_offset = global_h_pos - visible_cols + 2
        else:
            new_total_h_offset = total_h_offset

        # change visible scrollbars value
        widget.vscrollbar.setValue(new_total_v_offset)
        widget.hscrollbar.setValue(new_total_h_offset)

    def updateSectionHeight(self, logicalIndex, oldSize, newSize):
        self.setRowHeight(logicalIndex, newSize)

    def updateSectionWidth(self, logicalIndex, oldSize, newSize):
        self.setColumnWidth(logicalIndex, newSize)

    def autofit_columns(self):
        """Resize cells to contents"""
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        # Spyder loads more columns before resizing, but since it does not
        # load all columns anyway, I do not see the point
        # self.model().fetch_more_columns()
        self.resizeColumnsToContents()
        QApplication.restoreOverrideCursor()

    def updateGeometry(self):
        # Set maximum height
        if self.vpos == TOP:
            maximum_height = self.horizontalHeader().height() + \
                             sum(self.rowHeight(r) for r in range(self.model().rowCount()))
            self.setFixedHeight(maximum_height)
        # Set maximum width
        if self.hpos == LEFT:
            maximum_width = self.verticalHeader().width() + \
                            sum(self.columnWidth(c) for c in range(self.model().columnCount()))
            self.setFixedWidth(maximum_width)
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
                # FIXME: get minvalue & maxvalue from somewhere... the adapter? or the model?
                #  another specific adapter for minvalue, one for maxvalue, one for bg_value, etc.?
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
        model.setData(index, editor.currentText())


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

        self.context_menu = self.setup_context_menu()

        # TODO: find a cleaner way to do this
        # For some reason the shortcuts in the context menu are not available if the widget does not have the focus,
        # EVEN when using action.setShortcutContext(Qt.ApplicationShortcut) (or Qt.WindowShortcut) so we redefine them
        # here. I was also unable to get the function an action.triggered is connected to, so I couldn't do this via
        # a loop on self.context_menu.actions.
        shortcuts = [
            (keybinding('Copy'), self.parent().copy),
            (QKeySequence("Ctrl+E"), self.parent().to_excel),
            (keybinding('Paste'), self.parent().paste),
            (keybinding('Print'), self.parent().plot)
        ]
        for key_seq, target in shortcuts:
            shortcut = QShortcut(key_seq, self)
            shortcut.activated.connect(target)
        delegate = ArrayDelegate(self)
        self.setItemDelegate(delegate)

    def resizeEvent(self, event):
        assert isinstance(event, QResizeEvent)
        editor_widget = self.parent().parent()
        old_size = event.oldSize()
        new_size = event.size()
        print("resize", old_size, "->", new_size)
        if new_size.width() != old_size.width():
            editor_widget.hscrollbar.update_range()
        if new_size.height() != old_size.height():
            editor_widget.vscrollbar.update_range()
        AbstractView.resizeEvent(self, event)

    def selectRow(self, row: int):
        super().selectRow(row)
        total_v_size, total_h_size = self.model().adapter.shape2d()
        self.first_selection_corner = (row, 0)
        self.second_selection_corner = (row, total_h_size)

    def selectNewRow(self, row_index):
        # if not MultiSelection mode activated, selectRow will unselect previously
        # selected rows (unless SHIFT or CTRL key is pressed)

        # this produces a selection with multiple QItemSelectionRange. We could merge them here, but it is
        # easier to handle in selection_bounds
        self.setSelectionMode(QTableView.MultiSelection)
        # do not call self.selectRow to avoid updating first_selection_corner
        super().selectRow(row_index)
        total_v_size, total_h_size = self.model().adapter.shape2d()
        self.second_selection_corner = (row_index, total_h_size)
        self.setSelectionMode(QTableView.ContiguousSelection)

    def selectColumn(self, column: int):
        super().selectColumn(column)
        total_v_size, total_h_size = self.model().adapter.shape2d()
        self.first_selection_corner = (0, column)
        self.second_selection_corner = (total_v_size, column)

    def selectNewColumn(self, column_index):
        # if not MultiSelection mode activated, selectColumn will unselect previously
        # selected columns (unless SHIFT or CTRL key is pressed)

        # this produces a selection with multiple QItemSelectionRange. We could merge them here, but it is
        # easier to handle in selection_bounds
        self.setSelectionMode(QTableView.MultiSelection)
        self.selectColumn(column_index)
        self.setSelectionMode(QTableView.ContiguousSelection)

    def selectAll(self):
        super().selectAll()
        total_v_size, total_h_size = self.model().adapter.shape2d()
        self.first_selection_corner = (0, 0)
        self.second_selection_corner = (total_v_size, total_h_size)

    def setup_context_menu(self):
        """Setup context menu"""
        self.copy_action = create_action(self, _('Copy'),
                                         shortcut=keybinding('Copy'),
                                         icon=ima.icon('edit-copy'),
                                         triggered=lambda: self.signal_copy.emit())
        self.excel_action = create_action(self, _('Copy to Excel'),
                                          shortcut="Ctrl+E",
                                          # icon=ima.icon('edit-copy'),
                                          triggered=lambda: self.signal_excel.emit())
        self.paste_action = create_action(self, _('Paste'),
                                          shortcut=keybinding('Paste'),
                                          icon=ima.icon('edit-paste'),
                                          triggered=lambda: self.signal_paste.emit())
        self.plot_action = create_action(self, _('Plot'),
                                         shortcut=keybinding('Print'),
                                         # icon=ima.icon('editcopy'),
                                         triggered=lambda: self.signal_plot.emit())
        menu = QMenu(self)
        menu.addActions([self.copy_action, self.excel_action, self.plot_action, self.paste_action])
        return menu

    def contextMenuEvent(self, event):
        """Reimplement Qt method"""
        self.context_menu.popup(event.globalPos())
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Reimplement Qt method"""
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.first_selection_corner = self.get_cursor_global_pos()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Reimplement Qt method"""
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            self.second_selection_corner = self.get_cursor_global_pos()

    def keyPressEvent(self, event):
        """Reimplement Qt method"""

        # allow to start editing cells by pressing Enter
        if event.key() == Qt.Key_Return:
            index = self.currentIndex()
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
        selection_model = self.selectionModel()
        assert isinstance(selection_model, QItemSelectionModel)
        selection = selection_model.selection()
        assert isinstance(selection, QItemSelection)

        total_rows, total_cols = model.adapter.shape2d()
        if not selection:
            return 0, total_rows, 0, total_cols

        assert self.first_selection_corner is not None
        assert self.second_selection_corner is not None
        selection_v_pos1, selection_h_pos1 = self.first_selection_corner
        selection_v_pos2, selection_h_pos2 = self.second_selection_corner

        row_min = min(selection_v_pos1, selection_v_pos2)
        row_max = max(selection_v_pos1, selection_v_pos2)
        col_min = min(selection_h_pos1, selection_h_pos2)
        col_max = max(selection_h_pos1, selection_h_pos2)

        # FIXME: the local/Qt selection should be changed when scrolling !!!!

        # if the whole buffer row/column has been selected, select the whole dataset row/column
        # FIXME: this should not be done, but rather the "select row" and "select column" buttons should
        #        set both self.first_selection_corner and self.second_selection_corner (in addition to selecting
        #        the whole buffer like it does now)
        if row_min == 0 and row_max == model.nrows - 1:
            row_max = total_rows - 1
        if col_min == 0 and col_max == model.ncols - 1:
            col_max = total_cols - 1
        return row_min, row_max + 1, col_min, col_max + 1


def num_int_digits(value):
    """
    Number of integer digits. Completely ignores the fractional part. Does not take sign into account.

    >>> num_int_digits(1)
    1
    >>> num_int_digits(99)
    2
    >>> num_int_digits(-99.1)
    2
    """
    value = abs(value)
    log10 = math.log10(value) if value > 0 else 0
    if log10 == np.inf:
        return 308
    else:
        # max(1, ...) because there is at least one integer digit.
        # explicit conversion to int for Python2.x
        return max(1, int(math.floor(log10)) + 1)


class ScrollBar(QScrollBar):
    """
    A specialised scrollbar.
    """
    def __init__(self, parent, orientation, data_model, widget):
        super().__init__(orientation, parent)
        assert isinstance(data_model, DataArrayModel)

        self.model = data_model
        self.widget = widget

        # we need to update_range when the *total* number of rows/columns change, not when the loaded rows change
        # so connecting to the rowsInserted and columnsInserted signals is useless here
        data_model.modelReset.connect(self.update_range)

    def update_range(self):
        adapter = self.model.adapter
        if adapter is None:
            return
        # TODO: for some adapters shape2d is not reliable (it is a best guess), we should make sure we handle that
        total_rows, total_cols = adapter.shape2d()
        widget = self.widget
        if self.orientation() == Qt.Horizontal:
            visible_cols = widget.visible_cols()
            # FIXME: adding + 1 would be more correct (to account for partially visible cells) but it breaks other stuff
            max_value = total_cols - visible_cols
            print("horizontal update_range:", visible_cols, "visible cols =>", max_value, "maximum value")
        else:
            visible_rows = widget.visible_rows()
            # FIXME: adding + 1 would be more correct (to account for partially visible cells) but it breaks other stuff
            max_value = total_rows - visible_rows
            print("vertical update_range:", visible_rows, "visible rows =>", max_value, "maximum value")
        self.setMinimum(0)
        self.setMaximum(max_value)


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
        # font_metrics = QFontMetrics(self._used_font)
        # def str_width(c):
        #     return font_metrics.size(Qt.TextSingleLine, c).width()
        # return str_width
        return QFontMetrics(self._cached_font).width

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

    def __init__(self, parent, data=None, readonly=False, bg_value=None, bg_gradient='blue-red',
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

        # Synchronize resizing
        self.view_axes.horizontalHeader().sectionResized.connect(self.view_vlabels.updateSectionWidth)
        self.view_axes.verticalHeader().sectionResized.connect(self.view_hlabels.updateSectionHeight)
        self.view_hlabels.horizontalHeader().sectionResized.connect(self.view_data.updateSectionWidth)
        self.view_vlabels.verticalHeader().sectionResized.connect(self.view_data.updateSectionHeight)
        # Synchronize auto-resizing
        self.view_axes.horizontalHeader().sectionHandleDoubleClicked.connect(self.resize_axes_column_to_contents)
        self.view_hlabels.horizontalHeader().sectionHandleDoubleClicked.connect(self.resize_hlabels_column_to_contents)
        self.view_axes.verticalHeader().sectionHandleDoubleClicked.connect(self.resize_axes_row_to_contents)
        self.view_vlabels.verticalHeader().sectionHandleDoubleClicked.connect(self.resize_vlabels_row_to_contents)

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
        self.view_data.horizontalScrollBar().valueChanged.connect(self.view_hlabels.horizontalScrollBar().setValue)
        self.view_hlabels.horizontalScrollBar().valueChanged.connect(self.view_data.horizontalScrollBar().setValue)
        # data <--> vlabels
        self.view_data.verticalScrollBar().valueChanged.connect(self.view_vlabels.verticalScrollBar().setValue)
        self.view_vlabels.verticalScrollBar().valueChanged.connect(self.view_data.verticalScrollBar().setValue)

        # Synchronize selecting columns(rows) via hor.(vert.) header of x(y)labels view
        self.view_hlabels.horizontalHeader().sectionPressed.connect(self.view_data.selectColumn)
        self.view_hlabels.horizontalHeader().sectionEntered.connect(self.view_data.selectNewColumn)
        self.view_vlabels.verticalHeader().sectionPressed.connect(self.view_data.selectRow)
        self.view_vlabels.verticalHeader().sectionEntered.connect(self.view_data.selectNewRow)

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
        self.filters_layout = QHBoxLayout()
        self.btn_layout = QHBoxLayout()
        self.btn_layout.setAlignment(Qt.AlignLeft)

        # sometimes also called "Fractional digits" or "scale"
        label = QLabel("Decimal Places")
        self.btn_layout.addWidget(label)
        spin = QSpinBox(self)
        spin.valueChanged.connect(self.frac_digits_changed)
        self.digits_spinbox = spin
        self.btn_layout.addWidget(spin)
        self.frac_digits = 0

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
        layout.addLayout(self.filters_layout)
        layout.addWidget(array_frame)
        layout.addLayout(self.btn_layout)
        # left, top, right, bottom
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # set gradient
        self.model_data.set_bg_gradient(gradient_map[bg_gradient])

        # set data
        if data is not None:
            self.set_data(data, bg_value=bg_value, frac_digits=digits)

        # See http://doc.qt.io/qt-4.8/qt-draganddrop-fridgemagnets-dragwidget-cpp.html for an example
        self.setAcceptDrops(True)

    def visible_cols(self):
        """number of visible columns *including* partially visible ones"""

        view_data = self.view_data
        hidden_h_offset = view_data.horizontalScrollBar().value()
        last_col = view_data.columnAt(view_data.width() - 1)
        # +1 because last_col is a 0-based index
        # if last_col == -1 it means the visible area is larger than the array
        num_cols = last_col + 1 if last_col != -1 else self.model_data.ncols
        return num_cols - hidden_h_offset

    def visible_rows(self):
        """number of visible rows *including* partially visible ones"""

        view_data = self.view_data
        hidden_v_offset = view_data.verticalScrollBar().value()
        last_row = view_data.rowAt(view_data.height() - 1)
        # +1 because last_row is a 0-based index
        # if last_row == -1 it means the visible area is larger than the array
        num_rows = last_row + 1 if last_row != -1 else self.model_data.nrows
        return num_rows - hidden_v_offset

    def visible_vscroll_changed(self, value):
        # 'value' will be the first visible row
        assert value >= 0
        model_data = self.model_data
        hidden_vscroll = self.view_data.verticalScrollBar()
        v_offset = model_data.v_offset
        # invisible rows is the margin we have before we need to move the buffer
        # this is usually one less than hidden_vscroll.maximum() because of partially visible cells
        invisible_rows = model_data.nrows - self.visible_rows()
        assert invisible_rows >= 0, (model_data.nrows, self.visible_rows())
        extra_move = invisible_rows // 2

        print(f"visible vscroll changed (value: {value}, v_offset: {v_offset}, invis: {invisible_rows}, "
              f"extra_move: {extra_move})")

        # the buffer is beyond what we want to display, so we need to move it back
        if value < v_offset:
            # we could simply set it to value but we want to move more to avoid fetching data for each row
            new_v_offset = max(value - extra_move, 0)
            print("value < v_offset (min)", end=' ')

        # we don't need to move the buffer (we can absorb the scroll change entirely with the hidden scroll)
        elif value <= v_offset + invisible_rows:
            new_v_offset = v_offset
            print("min <= value <= max (hidden only)", end=' ')

        # the buffer is before what we want to display, so we need to move it further
        #           <-visible_rows->
        #        <------nrows---------->
        # |      |------buffer---------|    |       |          |
        # ^      ^                          ^       ^          ^
        # 0      v_offset                   value   max_value  total_rows
        else:
            # we could set it to "value - invisible_rows" to move as little as possible (this would place the visible
            # rows at the end of the buffer) but we want to move more to avoid fetching data each time we move a single
            # row
            new_v_offset = value - invisible_rows + extra_move
            print("value > v_offset + invis (max)", end=' ')

        assert new_v_offset >= 0
        assert new_v_offset <= value <= new_v_offset + invisible_rows

        new_hidden_offset = value - new_v_offset
        print(f"=> hidden = {new_hidden_offset}, new_v_offset = {new_v_offset}")
        if new_v_offset != v_offset:
            model_data.set_v_offset(new_v_offset)
            self.model_vlabels.set_v_offset(new_v_offset)
        hidden_vscroll.setValue(new_hidden_offset)

    def visible_hscroll_changed(self, value):
        # 'value' will be the first visible column
        assert value >= 0
        model_data = self.model_data
        hidden_hscroll = self.view_data.horizontalScrollBar()
        h_offset = model_data.h_offset
        # invisible cols is the margin we have before we need to move the buffer
        # this is usually one less than hidden_hscroll.maximum() because of partially visible cells
        invisible_cols = model_data.ncols - self.visible_cols()
        assert invisible_cols >= 0
        extra_move = invisible_cols // 2
        print(f"visible hscroll changed (value: {value}, h_offset: {h_offset}, invis: {invisible_cols}), "
              f"extra_move: {extra_move})")

        # the buffer is beyond what we want to display, so we need to move it back
        if value < h_offset:
            # we could simply set it to value but we want to move more to avoid fetching data for each row
            new_h_offset = max(value - extra_move, 0)
            print("value < h_offset (min)", end=' ')

        # we don't need to move the buffer (we can absorb the scroll change entirely with the hidden scroll)
        elif value <= h_offset + invisible_cols:
            new_h_offset = h_offset
            print("min <= value <= max (hidden only)", end=' ')

        # the buffer is before what we want to display, so we need to move it further
        #           <-visible_cols->
        #        <------ncols---------->
        # |      |------buffer---------|    |       |          |
        # ^      ^                          ^       ^          ^
        # 0      h_offset                   value   max_value  total_cols
        else:
            # we could set it to "value - invisible_cols" to move as little as possible (this would place the visible
            # cols at the end of the buffer) but we want to move more to avoid fetching data each time we move a single
            # col
            new_h_offset = value - invisible_cols + extra_move
            print("value > h_offset + invis (max)", end=' ')

        assert new_h_offset >= 0
        assert new_h_offset <= value <= new_h_offset + invisible_cols

        new_hidden_offset = value - new_h_offset
        print(f"=> hidden = {new_hidden_offset}, new_h_offset = {new_h_offset}")
        if new_h_offset != h_offset:
            model_data.set_h_offset(new_h_offset)
            self.model_hlabels.set_h_offset(new_h_offset)
        hidden_hscroll.setValue(new_hidden_offset)

    def gradient_changed(self, index):
        gradient = self.gradient_chooser.itemData(index) if index > 0 else None
        self.model_data.set_bg_gradient(gradient)

    def data_changed(self, data_model_changes):
        global_changes = self.data_adapter.translate_changes(data_model_changes)
        self.dataChanged.emit(global_changes)

    def mousePressEvent(self, event):
        # this method is not triggered at all when clicking on any of the tableviews
        self.dragLabel = self.childAt(event.pos()) if event.button() == Qt.LeftButton else None
        self.dragStartPosition = event.pos()

    def mouseMoveEvent(self, event):
        from qtpy.QtCore import QMimeData, QByteArray
        from qtpy.QtGui import QPixmap, QDrag

        if not (event.button() != Qt.LeftButton and isinstance(self.dragLabel, QLabel)):
            return

        if (event.pos() - self.dragStartPosition).manhattanLength() < QApplication.startDragDistance():
            return

        axis_index = self.filters_layout.indexOf(self.dragLabel) // 2

        # prepare hotSpot, mimeData and pixmap objects
        mimeData = QMimeData()
        mimeData.setText(self.dragLabel.text())
        mimeData.setData("application/x-axis-index", QByteArray.number(axis_index))
        pixmap = QPixmap(self.dragLabel.size())
        self.dragLabel.render(pixmap)

        # prepare drag object
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos() - self.dragStartPosition)

        drag.exec_(Qt.MoveAction | Qt.CopyAction, Qt.CopyAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            if self.filters_layout.geometry().contains(event.pos()):
                event.setDropAction(Qt.MoveAction)
                event.accept()
            else:
                event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText() and self.filters_layout.geometry().contains(event.pos()):
            child = self.childAt(event.pos())
            if isinstance(child, QLabel) and child.text() != "Filters":
                event.setDropAction(Qt.MoveAction)
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasText():
            if self.filters_layout.geometry().contains(event.pos()):
                old_index, success = event.mimeData().data("application/x-axis-index").toInt()
                new_index = self.filters_layout.indexOf(self.childAt(event.pos())) // 2

                data, bg_value = self.data_adapter.data, self.data_adapter.bg_value
                data, bg_value = self.data_adapter.move_axis(data, bg_value, old_index, new_index)
                self.set_data(data, bg_value)

                event.setDropAction(Qt.MoveAction)
                event.accept()
            else:
                event.acceptProposedAction()
        else:
            event.ignore()

    def _update_models(self, reset_model_data):
        self.model_axes.set_adapter(self.data_adapter)
        self.model_hlabels.set_adapter(self.data_adapter)
        self.model_vlabels.set_adapter(self.data_adapter)
        self.model_data.set_adapter(self.data_adapter)
        # # bg value
        # # use flag reset=False to avoid calling reset() several times
        # bg_value = self.data_adapter.get_bg_value()
        # self.model_data.set_bg_value(bg_value, reset=False)
        # # reset the data model if required
        # if reset_model_data:
        #     self.model_data.reset()

    def set_data(self, data, bg_value=None, frac_digits=None):
        # get new adapter instance + set data
        # TODO: pass a dict to get_adapter {'data': data, 'bg_value': bg_value}
        # TODO: add a mechanism that adapters can use to tell whether they support a
        #       particular instance of a data structure. This should probably be a class method.
        #       For example for memoryview, "structured"
        #       memoryview are not supported and get_adapter currently returns None
        self.data_adapter = get_adapter(data=data, bg_value=bg_value)
        # reset scrollbars
        self.vscrollbar.setValue(0)
        self.hscrollbar.setValue(0)
        # update filters
        self._reset_filter_bar()
        # update models
        # Note: model_data is reset by call of set_format below
        self._update_models(reset_model_data=False)
        # reset default size
        self._reset_default_size()
        # update data format
        self.set_format(frac_digits=frac_digits, scientific=None)
        # update gradient_chooser
        # FIXME: implement a bgcolor_possible property in the adapter
        # self.gradient_chooser.setEnabled(self.data_adapter.bgcolor_possible)

    def _reset_default_size(self):
        self.view_axes.set_default_size()
        self.view_vlabels.set_default_size()
        self.view_hlabels.set_default_size()
        self.view_data.set_default_size()

    def _reset_filter_bar(self):
        filters_layout = self.filters_layout
        clear_layout(filters_layout)
        filters = self.data_adapter.get_filters()
        # size > 0 to avoid arrays with length 0 axes and len(axes) > 0 to avoid scalars (scalar.size == 1)
        if filters: #self.data_adapter.size > 0 and len(filters) > 0:
            filters_layout.addWidget(QLabel(_("Filters")))
            for filter_idx, (filter_name, filter_labels) in enumerate(filters):
                filters_layout.addWidget(QLabel(filter_name))
                # FIXME: on very large axes, this is getting too slow. Ideally the combobox should use a model which
                # only fetch labels when they are needed to be displayed
                if len(filter_labels) < 10000:
                    filters_layout.addWidget(self.create_filter_combo(filter_idx, filter_name, filter_labels))
                else:
                    filters_layout.addWidget(QLabel("too big to be filtered"))
            filters_layout.addStretch()

    def set_format(self, frac_digits=None, scientific=None):
        """Set format.

        Parameters
        ----------
        frac_digits : int, optional
            Number of decimals to display. Defaults to None (autodetect).
        scientific : boolean, optional
            Whether or not to display values in scientific format. Defaults to None (autodetect).
        """
        assert frac_digits is None or isinstance(frac_digits, int)
        assert scientific is None or isinstance(scientific, bool)
        scientific_toggled = scientific is not None and scientific != self.use_scientific
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"ArrayEditorWidget.set_format(frac_digits={frac_digits}, scientific={scientific})")

        data_sample = self.data_adapter.get_sample()
        is_number_dtype = isinstance(data_sample, np.ndarray) and np.issubdtype(data_sample.dtype, np.number)
        cur_colwidth = self._get_current_min_col_width()

        # TODO: we should also compute column width for text/object dtype
        if is_number_dtype and data_sample.size:
            # TODO: vmin/vmax should come from the adapter (were it is already computed!!!)
            #       (but modified whenever the data changes)
            # TODO: some (all?) of this should be done in the adapter because it knows whether vmin/vmax should be per
            #       column or global and in the end if format and colwidth should be the same for the whole
            #       array or per col but I am still unsure of the boundary because font_metrics should not be used in
            #       the adapter.
            #       The adapter also knows how expensive it is to compute some stuff and whether we can compute vmin/
            #       vmax on the full array or have to rely on sample + "rolling" vmin/vmax.

            #       I guess the widget should ask the adapter (or the model?) how many characters there is in each
            #       cell. But since the model would not know how to answer that, Maybe
            vmin, vmax = np.min(data_sample), np.max(data_sample)
            is_finite_data = np.isfinite(vmin) and np.isfinite(vmax)
            if is_finite_data:
                finite_vmin, finite_vmax = vmin, vmax
                finite_sample = data_sample
            else:
                isfinite = np.isfinite(data_sample)
                if isfinite.any():
                    finite_sample = data_sample[isfinite]
                    finite_vmin, finite_vmax = np.min(finite_sample), np.max(finite_sample)
                else:
                    scientific = False
                    frac_digits = 0
                    finite_vmin, finite_vmax = 0, 0

            int_digits = max(num_int_digits(finite_vmin), num_int_digits(finite_vmax))
            if not is_finite_data:
                # so that we have enough room to display "nan" or "inf"
                # ideally we should add a finite_data argument to get_numbers_width so that we take the actual
                # "nan" and "inf" strings width but I am unsure it is worth it
                int_digits = max(int_digits, 3)
            has_negative = finite_vmin < 0

            font_metrics = self.font_metrics

            # choose whether or not to use scientific notation
            # ================================================
            if scientific is None:
                # use scientific format if there are more integer digits than we can display or if we can display
                # more information that way (scientific format "uses" 4 digits, so we have a net win if we have
                # >= 4 zeros -- *including the integer one*)
                # TODO: only do so if we would actually display more information
                # 0.00001 can be displayed with 8 chars
                # 1e-05
                # would
                absmax = max(abs(finite_vmin), abs(finite_vmax))
                logabsmax = math.log10(absmax) if absmax else 0
                # minimum number of zeros before meaningful fractional part
                frac_zeros = math.ceil(-logabsmax) - 1 if logabsmax < 0 else 0
                non_scientific_int_width = font_metrics.get_numbers_width(int_digits, need_sign=has_negative)
                scientific = non_scientific_int_width > cur_colwidth or frac_zeros >= 4
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f" -> detected scientific={scientific}")

            # determine best number of decimals to display
            # ============================================
            if frac_digits is None:
                int_part_width = font_metrics.get_numbers_width(int_digits, need_sign=has_negative,
                                                                scientific=scientific)
                # since we are computing the number of frac digits, we always need the dot
                avail_width_for_frac_part = max(cur_colwidth - int_part_width - font_metrics.dot_width, 0)
                max_frac_digits = avail_width_for_frac_part // font_metrics.digit_width
                frac_digits = self._data_frac_digits(finite_sample, max_frac_digits=max_frac_digits)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f" -> detected frac_digits={frac_digits}")

            format_letter = 'e' if scientific else 'f'
            fmt = '%%.%d%s' % (frac_digits, format_letter)
            data_colwidth = font_metrics.get_numbers_width(int_digits, frac_digits, need_sign=has_negative,
                                                           scientific=scientific)
        else:
            frac_digits = 0
            scientific = False
            fmt = '%s'
            # TODO: compute actual column width using data
            # this is done below but this code would benefit from a cleanup
            data_colwidth = 60

        self.data_adapter.set_format(fmt)
        self.model_data._get_data()
        self.model_data.reset()

        self.frac_digits = frac_digits
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

        # frac digits changed => set new column width
        if not scientific_toggled or data_colwidth > cur_colwidth:
            header = self.view_hlabels.horizontalHeader()

            hlabels = self.model_hlabels.processed_data[Qt.DisplayRole]
            data_strings = self.model_data.processed_data[Qt.DisplayRole]

            num_buffer_hlabels = len(hlabels[0])

            # FIXME: this will set width of the 40 first columns (otherwise it gets very slow, eg. big1d)
            #        now that the buffer branch is merged we should probably *set* widths for all visible columns
            #        BUT we also need to keep track of the widths of ALL columns (including hidden ones) and
            #        *resize* columns when moving the viewport (aka scrolling using the visible scrollbars)
            #        so that
            #        1) column widths are correct for columns past the buffer size
            #        2) user changed columns width are not lost when scrolling back & forth
            #        I guess we will have to compute column width as we go (when we first see a column)
            num_cols = min(header.count(), num_buffer_hlabels, 40)
            str_width = FontMetrics(self.model_hlabels).str_width

            MIN_COLWITH = 30

            MARGIN_WIDTH = 8  # empirically measured
            data_inner_colwidth = max(data_colwidth, MIN_COLWITH) - MARGIN_WIDTH
            sample_size = data_sample.size if isinstance(data_sample, np.ndarray) else len(data_sample)

            for i in range(num_cols):
                max_header_width = max(str_width(row_labels[i]) for row_labels in hlabels)
                if sample_size:
                    if is_number_dtype:
                        max_data_width = data_inner_colwidth
                    else:
                        max_data_width = max(str_width(row_data[i]) for row_data in data_strings)
                else:
                    max_data_width = MIN_COLWITH - MARGIN_WIDTH
                colwidth = MARGIN_WIDTH + max(max_header_width, max_data_width)
                header.resizeSection(i, colwidth)

    def _get_current_min_col_width(self):
        header = self.view_hlabels.horizontalHeader()
        if header.count():
            return min(header.sectionSize(i) for i in range(header.count()))
        else:
            return 0

    def _data_frac_digits(self, data, max_frac_digits):
        if not data.size:
            return 0
        threshold = 10 ** -(max_frac_digits + 1)
        for frac_digits in range(max_frac_digits):
            maxdiff = np.max(np.abs(data - np.round(data, frac_digits)))
            if maxdiff < threshold:
                return frac_digits
        return max_frac_digits

    def autofit_columns(self):
        self.view_axes.autofit_columns()
        for column in range(self.model_axes.columnCount()):
            self.resize_axes_column_to_contents(column)
        self.view_hlabels.autofit_columns()
        for column in range(self.model_hlabels.columnCount()):
            self.resize_hlabels_column_to_contents(column)

    def resize_axes_column_to_contents(self, column):
        # must be connected to view_axes.horizontalHeader().sectionHandleDoubleClicked signal
        width = max(self.view_axes.horizontalHeader().sectionSize(column),
                    self.view_vlabels.sizeHintForColumn(column))
        # no need to call resizeSection on view_vlabels (see synchronization lines in init)
        self.view_axes.horizontalHeader().resizeSection(column, width)

    def resize_hlabels_column_to_contents(self, column):
        # must be connected to view_labels.horizontalHeader().sectionHandleDoubleClicked signal
        width = max(self.view_hlabels.horizontalHeader().sectionSize(column),
                    self.view_data.sizeHintForColumn(column))
        # no need to call resizeSection on view_data (see synchronization lines in init)
        self.view_hlabels.horizontalHeader().resizeSection(column, width)

    def resize_axes_row_to_contents(self, row):
        # must be connected to view_axes.verticalHeader().sectionHandleDoubleClicked
        height = max(self.view_axes.verticalHeader().sectionSize(row),
                     self.view_hlabels.sizeHintForRow(row))
        # no need to call resizeSection on view_hlabels (see synchronization lines in init)
        self.view_axes.verticalHeader().resizeSection(row, height)

    def resize_vlabels_row_to_contents(self, row):
        # must be connected to view_labels.verticalHeader().sectionHandleDoubleClicked
        height = max(self.view_vlabels.verticalHeader().sectionSize(row),
                     self.view_data.sizeHintForRow(row))
        # no need to call resizeSection on view_data (see synchronization lines in init)
        self.view_vlabels.verticalHeader().resizeSection(row, height)

    def scientific_changed(self, value):
        # auto-detect frac_digits
        self.set_format(frac_digits=None, scientific=bool(value))

    def frac_digits_changed(self, value):
        self.set_format(value, self.use_scientific)

    def change_filter(self, filter_idx, filter_name, indices):
        self.data_adapter.update_filter(filter_idx, filter_name, indices)
        # FIXME: we should implement something lighter in this case (the adapter is the same anyway)
        self._update_models(reset_model_data=True)

    def create_filter_combo(self, filter_idx, filter_name, filter_labels):
        def filter_changed(checked_items):
            self.change_filter(filter_idx, filter_name, checked_items)
        combo = FilterComboBox(self)
        combo.addItems([str(label) for label in filter_labels])
        combo.checkedItemsChanged.connect(filter_changed)
        return combo

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
            QMessageBox.critical(self, "Error", "to_excel() is not available because xlwings is not installed")

    def paste(self):
        # FIXME: this now returns coordinates in global space while the rest of this function assumes local/buffer
        #        space coordinates. But this whole "set_values" code should be revisited entirely anyway
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

        # FIXME: the way to change data is very convoluted:
        #        either Widget.paste or ArrayDelegate.setModelData (via model.setData) calls model.set_values
        #        on the model but that does not
        #        change any data but computes a {global_filtered_2D_coords: old_value, new_value) dict of changes
        #        and emits a newChanges signal and then a dataChanged (builtin) signal (which only makes senses because
        #        a signal .emit call only returns when all its slots have executed, hence the whole chain below has
        #        already been executed when that second signal is emitted).
        #        then the newChanges signal is caught by the widget, which asks the
        #        adapter to transform the changes from 2d global (but potentially filtered) positional keys
        #        to ND global positional keys, then re-emits a dataChanged signal with a list of those changes,
        #        the editor catches that signal and push those changes to the edit_undo_stack which actually
        #        applies each change by using
        #           kernel.shell.run_cell(f"{self.target}.i[{key}] = {new_value}")
        #           OR
        #           self.target.i[key] = new_value
        #        and there, editor.arraywidget.model_data.reset() is called which notifies qt the whole thing
        #        needs to be refreshed (including reprocessing the data via _process_data but does not fetch the actual
        #        new data!!!)
        #        and it actually only appears to work in the simple case of editing an unfiltered array because
        #        we are using array *views* all the way so when we edit the array, the "raw_data" in the model is
        #        updated directly too and _process_data is indeed enough.

        #        Since we can *NOT* push a command on the edit_undo_stack without executing it, we should:
        #        * create widget.set_values method, call it from paste a the ArrayDelegate
        #        * create NDkey: changes (by asking adapterask the adapter calldirectly create command
        #        * directly create command

        #        as a side note the (visible) Scrollbar is connected to the reset event and updates its range in that
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
        from larray_editor.editor import AbstractEditor, MappingEditor
        try:
            figure = self.data_adapter.plot(*self.view_data.selection_bounds())
            widget = self
            while widget is not None and not isinstance(widget, AbstractEditor) and callable(widget.parent):
                widget = widget.parent()
            title = widget.current_array_name if isinstance(widget, MappingEditor) else None
            show_figure(self, figure, title)
        except ImportError:
            QMessageBox.critical(self, "Error", "plot() is not available because matplotlib is not installed")
