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
# it does seem to be really designed for very large arrays and it would
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

from __future__ import print_function

import math
from itertools import chain

import numpy as np
from qtpy.QtCore import Qt, QPoint, QItemSelection, QItemSelectionModel, Signal, QSize
from qtpy.QtGui import (QDoubleValidator, QIntValidator, QKeySequence, QFontMetrics, QCursor, QPixmap, QPainter,
                        QLinearGradient, QColor, QIcon)
from qtpy.QtWidgets import (QApplication, QTableView, QItemDelegate, QLineEdit, QCheckBox,
                            QMessageBox, QMenu, QLabel, QSpinBox, QWidget, QToolTip, QShortcut, QScrollBar,
                            QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy, QFrame, QComboBox)

try:
    import xlwings as xw
except ImportError:
    xw = None

from larray_editor.utils import (keybinding, create_action, clear_layout, get_font, from_qvariant, to_qvariant,
                                 is_number, is_float, _, ima, LinearGradient)
from larray_editor.arrayadapter import LArrayDataAdapter
from larray_editor.arraymodel import LabelsArrayModel, DataArrayModel
from larray_editor.combo import FilterComboBox, FilterMenu
import larray as la

# XXX: define Enum instead ?
TOP, BOTTOM = 0, 1
LEFT, RIGHT = 0, 1

class LabelsView(QTableView):
    """"Labels view class"""

    allSelected = Signal()

    def __init__(self, parent, model, position):
        QTableView.__init__(self, parent)
        # set model
        if not isinstance(model, LabelsArrayModel):
            raise TypeError("Expected model of type {}. Received {} instead"
                            .format(LabelsArrayModel.__name__, type(model).__name__))
        self.setModel(model)
        # set position
        if not (isinstance(position, (list, tuple)) and len(position) == 2):
            raise TypeError("Expected tuple or list of length 2")
        self.position = position

        self.setSelectionMode(QTableView.ContiguousSelection)

        self.horizontalHeader().setFrameStyle(QFrame.NoFrame)
        self.verticalHeader().setFrameStyle(QFrame.NoFrame)

        self.set_default_size()
        # to fetch more rows/columns when required
        self.horizontalScrollBar().valueChanged.connect(self.on_horizontal_scroll_changed)
        self.verticalScrollBar().valueChanged.connect(self.on_vertical_scroll_changed)

        # hide horizontal/vertical headers
        if position == (TOP, RIGHT):
            self.verticalHeader().hide()
        elif position == (BOTTOM, LEFT):
            self.horizontalHeader().hide()

        # Hide scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.model().modelReset.connect(self.updateGeometry)
        self.horizontalHeader().sectionResized.connect(self.updateGeometry)
        self.verticalHeader().sectionResized.connect(self.updateGeometry)

    def set_default_size(self):
        # make the grid a bit more compact
        self.horizontalHeader().setDefaultSectionSize(64)
        self.horizontalHeader().setFixedHeight(10)
        self.verticalHeader().setDefaultSectionSize(20)
        self.verticalHeader().setFixedWidth(10)

    def on_vertical_scroll_changed(self, value):
        if value == self.verticalScrollBar().maximum():
            self.model().fetch_more_rows()

    def on_horizontal_scroll_changed(self, value):
        if value == self.horizontalScrollBar().maximum():
            self.model().fetch_more_columns()

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

    def selectAll(self):
        self.allSelected.emit()

    def updateGeometry(self):
        # Set maximum height
        if self.position[0] == TOP:
            maximum_height = self.horizontalHeader().height() + \
                             sum(self.rowHeight(r) for r in range(self.model().rowCount()))
            self.setFixedHeight(maximum_height)
        # Set maximum width
        if self.position[1] == LEFT:
            maximum_width = self.verticalHeader().width() + \
                            sum(self.columnWidth(c) for c in range(self.model().columnCount()))
            self.setFixedWidth(maximum_width)
        # update geometry
        super().updateGeometry()


class ArrayDelegate(QItemDelegate):
    """Array Editor Item Delegate"""
    def __init__(self, dtype, parent=None, font=None,
                 minvalue=None, maxvalue=None):
        QItemDelegate.__init__(self, parent)
        self.dtype = dtype
        if font is None:
            font = get_font('arrayeditor')
        self.font = font
        self.minvalue = minvalue
        self.maxvalue = maxvalue

        # We must keep a count instead of the "current" one, because when
        # switching from one cell to the next, the new editor is created
        # before the old one is destroyed, which means it would be set to None
        # when the old one is destroyed.
        self.editor_count = 0

    def createEditor(self, parent, option, index):
        """Create editor widget"""
        model = index.model()
        value = model.get_value(index)
        if self.dtype.name == "bool":
            # toggle value
            value = not value
            model.setData(index, to_qvariant(value))
            return
        elif value is not np.ma.masked:
            minvalue, maxvalue = self.minvalue, self.maxvalue
            if minvalue is not None and maxvalue is not None:
                msg = "value must be between %s and %s" % (minvalue, maxvalue)
            elif minvalue is not None:
                msg = "value must be >= %s" % minvalue
            elif maxvalue is not None:
                msg = "value must be <= %s" % maxvalue
            else:
                msg = None

            # Not using a QSpinBox for integer inputs because I could not find
            # a way to prevent the spinbox/editor from closing if the value is
            # invalid. Using the builtin minimum/maximum of the spinbox works
            # but that provides no message so it is less clear.
            editor = QLineEdit(parent)
            if is_number(self.dtype):
                validator = QDoubleValidator(editor) if is_float(self.dtype) \
                    else QIntValidator(editor)
                if minvalue is not None:
                    validator.setBottom(minvalue)
                if maxvalue is not None:
                    validator.setTop(maxvalue)
                editor.setValidator(validator)

                def on_editor_text_edited():
                    if not editor.hasAcceptableInput():
                        QToolTip.showText(editor.mapToGlobal(QPoint()), msg)
                    else:
                        QToolTip.hideText()
                if msg is not None:
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
        text = from_qvariant(index.model().data(index, Qt.DisplayRole), str)
        editor.setText(text)


class DataView(QTableView):
    """Data array view class"""

    signal_copy = Signal()
    signal_excel = Signal()
    signal_paste = Signal()
    signal_plot = Signal()

    def __init__(self, parent, model, dtype, shape):
        QTableView.__init__(self, parent)
        # set model
        if not isinstance(model, DataArrayModel):
            raise TypeError("Expected model of type {}. Received {} instead"
                            .format(DataArrayModel.__name__, type(model).__name__))
        self.setModel(model)
        # set array delegate
        delegate = ArrayDelegate(dtype, self, minvalue=model.minvalue, maxvalue=model.maxvalue)
        self.setItemDelegate(delegate)

        self.setSelectionMode(QTableView.ContiguousSelection)

        self.shape = shape
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

        self.horizontalHeader().setFrameStyle(QFrame.NoFrame)
        self.verticalHeader().setFrameStyle(QFrame.NoFrame)

        self.set_default_size()
        # Hide horizontal+vertical headers
        self.horizontalHeader().hide()
        self.verticalHeader().hide()

        # Hide scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # to fetch more rows/columns when required
        self.horizontalScrollBar().valueChanged.connect(self.on_horizontal_scroll_changed)
        self.verticalScrollBar().valueChanged.connect(self.on_vertical_scroll_changed)

        # self.horizontalHeader().sectionClicked.connect(self.on_horizontal_header_clicked)

    def set_default_size(self):
        # make the grid a bit more compact
        self.horizontalHeader().setDefaultSectionSize(64)
        self.verticalHeader().setDefaultSectionSize(20)

    # def on_horizontal_header_clicked(self, section_index):
    #     menu = FilterMenu(self)
    #     header = self.horizontalHeader()
    #     headerpos = self.mapToGlobal(header.pos())
    #     posx = headerpos.x() + header.sectionPosition(section_index)
    #     posy = headerpos.y() + header.height()
    #     menu.exec_(QPoint(posx, posy))

    def on_vertical_scroll_changed(self, value):
        if value == self.verticalScrollBar().maximum():
            self.model().fetch_more_rows()

    def on_horizontal_scroll_changed(self, value):
        if value == self.horizontalScrollBar().maximum():
            self.model().fetch_more_columns()

    def updateSectionHeight(self, logicalIndex, oldSize, newSize):
        self.setRowHeight(logicalIndex, newSize)

    def updateSectionWidth(self, logicalIndex, oldSize, newSize):
        self.setColumnWidth(logicalIndex, newSize)

    def selectNewRow(self, row_index):
        # if not MultiSelection mode activated, selectRow will unselect previously
        # selected rows (unless SHIFT or CTRL key is pressed)

        # this produces a selection with multiple QItemSelectionRange. We could merge them here, but it is
        # easier to handle in _selection_bounds
        self.setSelectionMode(QTableView.MultiSelection)
        self.selectRow(row_index)
        self.setSelectionMode(QTableView.ContiguousSelection)

    def selectNewColumn(self, column_index):
        # if not MultiSelection mode activated, selectColumn will unselect previously
        # selected columns (unless SHIFT or CTRL key is pressed)

        # this produces a selection with multiple QItemSelectionRange. We could merge them here, but it is
        # easier to handle in _selection_bounds
        self.setSelectionMode(QTableView.MultiSelection)
        self.selectColumn(column_index)
        self.setSelectionMode(QTableView.ContiguousSelection)

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

    def autofit_columns(self):
        """Resize cells to contents"""
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        # Spyder loads more columns before resizing, but since it does not
        # load all columns anyway, I do not see the point
        # self.model().fetch_more_columns()
        self.resizeColumnsToContents()
        QApplication.restoreOverrideCursor()

    def contextMenuEvent(self, event):
        """Reimplement Qt method"""
        self.context_menu.popup(event.globalPos())
        event.accept()

    def keyPressEvent(self, event):
        """Reimplement Qt method"""

        # comparing with the keysequence and not with event directly as we
        # did before because that only seems to work for shortcut
        # defined using QKeySequence.StandardKey, which is not the case for
        # Ctrl + E
        keyseq = QKeySequence(event.modifiers() | event.key())
        if keyseq == QKeySequence.Copy:
            self.copy()
        elif keyseq == QKeySequence.Paste:
            self.paste()
        elif keyseq == QKeySequence.Print:
            self.parent().plot()
        elif keyseq == "Ctrl+E":
            self.to_excel()
        # allow to start editing cells by pressing Enter
        elif event.key() == Qt.Key_Return and not self.model().readonly:
            index = self.currentIndex()
            if self.itemDelegate(index).editor_count == 0:
                self.edit(index)
        else:
            QTableView.keyPressEvent(self, event)

    def _selection_bounds(self, none_selects_all=True):
        """
        Parameters
        ----------
        none_selects_all : bool, optional
            If True (default) and selection is empty, returns all data.

        Returns
        -------
        tuple
            selection bounds. end bound is exclusive
        """
        model = self.model()
        selection_model = self.selectionModel()
        assert isinstance(selection_model, QItemSelectionModel)
        selection = selection_model.selection()
        assert isinstance(selection, QItemSelection)
        if not selection:
            if none_selects_all:
                return 0, model.total_rows, 0, model.total_cols
            else:
                return None
        # merge potentially multiple selections into one big rect
        row_min = min(srange.top() for srange in selection)
        row_max = max(srange.bottom() for srange in selection)
        col_min = min(srange.left() for srange in selection)
        col_max = max(srange.right() for srange in selection)

        # if not all rows/columns have been loaded
        if row_min == 0 and row_max == self.model().rows_loaded - 1:
            row_max = self.model().total_rows - 1
        if col_min == 0 and col_max == self.model().cols_loaded - 1:
            col_max = self.model().total_cols - 1
        return row_min, row_max + 1, col_min, col_max + 1


def ndigits(value):
    """
    number of integer digits

    >>> ndigits(1)
    1
    >>> ndigits(99)
    2
    >>> ndigits(-99.1)
    3
    """
    negative = value < 0
    value = abs(value)
    log10 = math.log10(value) if value > 0 else 0
    if log10 == np.inf:
        int_digits = 308
    else:
        # max(1, ...) because there is at least one integer digit.
        # explicit conversion to int for Python2.x
        int_digits = max(1, int(math.floor(log10)) + 1)
    # one digit for sign if negative
    return int_digits + negative


class ScrollBar(QScrollBar):
    """
    A specialised scrollbar.
    """
    def __init__(self, parent, data_scrollbar):
        super(ScrollBar, self).__init__(data_scrollbar.orientation(), parent)
        self.setMinimum(data_scrollbar.minimum())
        self.setMaximum(data_scrollbar.maximum())
        self.setSingleStep(data_scrollbar.singleStep())
        self.setPageStep(data_scrollbar.pageStep())

        data_scrollbar.valueChanged.connect(self.setValue)
        self.valueChanged.connect(data_scrollbar.setValue)

        data_scrollbar.rangeChanged.connect(self.setRange)
        self.rangeChanged.connect(data_scrollbar.setRange)


available_gradients = [
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



class ArrayEditorWidget(QWidget):
    def __init__(self, parent, data=None, readonly=False, bg_value=None, bg_gradient='blue-red',
                 minvalue=None, maxvalue=None):
        QWidget.__init__(self, parent)
        assert bg_gradient in gradient_map
        readonly = np.isscalar(data)
        self.readonly = readonly

        self.model_axes = LabelsArrayModel(parent=self, readonly=readonly)
        self.view_axes = LabelsView(parent=self, model=self.model_axes, position=(TOP, LEFT))

        self.model_xlabels = LabelsArrayModel(parent=self, readonly=readonly)
        self.view_xlabels = LabelsView(parent=self, model=self.model_xlabels, position=(TOP, RIGHT))

        self.model_ylabels = LabelsArrayModel(parent=self, readonly=readonly)
        self.view_ylabels = LabelsView(parent=self, model=self.model_ylabels, position=(BOTTOM, LEFT))

        self.model_data = DataArrayModel(parent=self, readonly=readonly, minvalue=minvalue, maxvalue=maxvalue)
        self.view_data = DataView(parent=self, model=self.model_data, dtype=data.dtype, shape=data.shape)

        self.data_adapter = LArrayDataAdapter(axes_model=self.model_axes, xlabels_model=self.model_xlabels,
                                              ylabels_model=self.model_ylabels, data_model=self.model_data, data=data,
                                              bg_value=bg_value, bg_gradient=bg_gradient)

        # Create vertical and horizontal scrollbars
        self.vscrollbar = ScrollBar(self, self.view_data.verticalScrollBar())
        self.hscrollbar = ScrollBar(self, self.view_data.horizontalScrollBar())

        # Synchronize resizing
        self.view_axes.horizontalHeader().sectionResized.connect(self.view_ylabels.updateSectionWidth)
        self.view_axes.verticalHeader().sectionResized.connect(self.view_xlabels.updateSectionHeight)
        self.view_xlabels.horizontalHeader().sectionResized.connect(self.view_data.updateSectionWidth)
        self.view_ylabels.verticalHeader().sectionResized.connect(self.view_data.updateSectionHeight)
        # Synchronize auto-resizing
        self.view_axes.horizontalHeader().sectionHandleDoubleClicked.connect(self.resize_axes_column_to_contents)
        self.view_xlabels.horizontalHeader().sectionHandleDoubleClicked.connect(self.resize_xlabels_column_to_contents)
        self.view_axes.verticalHeader().sectionHandleDoubleClicked.connect(self.resize_axes_row_to_contents)
        self.view_ylabels.verticalHeader().sectionHandleDoubleClicked.connect(self.resize_ylabels_row_to_contents)

        # synchronize specific methods
        self.view_axes.allSelected.connect(self.view_data.selectAll)
        self.view_data.signal_copy.connect(self.copy)
        self.view_data.signal_excel.connect(self.to_excel)
        self.view_data.signal_paste.connect(self.paste)
        self.view_data.signal_plot.connect(self.plot)

        # Synchronize scrolling
        # data <--> xlabels
        self.view_data.horizontalScrollBar().valueChanged.connect(self.view_xlabels.horizontalScrollBar().setValue)
        self.view_xlabels.horizontalScrollBar().valueChanged.connect(self.view_data.horizontalScrollBar().setValue)
        # data <--> ylabels
        self.view_data.verticalScrollBar().valueChanged.connect(self.view_ylabels.verticalScrollBar().setValue)
        self.view_ylabels.verticalScrollBar().valueChanged.connect(self.view_data.verticalScrollBar().setValue)

        # Synchronize selecting columns(rows) via hor.(vert.) header of x(y)labels view
        self.view_xlabels.horizontalHeader().sectionPressed.connect(self.view_data.selectColumn)
        self.view_xlabels.horizontalHeader().sectionEntered.connect(self.view_data.selectNewColumn)
        self.view_ylabels.verticalHeader().sectionPressed.connect(self.view_data.selectRow)
        self.view_ylabels.verticalHeader().sectionEntered.connect(self.view_data.selectNewRow)

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
        self.view_xlabels.setFrameStyle(QFrame.NoFrame)
        self.view_ylabels.setFrameStyle(QFrame.NoFrame)
        self.view_data.setFrameStyle(QFrame.NoFrame)
        # Set layout of table views:
        # [ axes  ][xlabels]|V|
        # [ylabels][ data  ]|s|
        # |  H. scrollbar  |
        array_layout = QGridLayout()
        array_layout.addWidget(self.view_axes, 0, 0)
        array_layout.addWidget(self.view_xlabels, 0, 1)
        array_layout.addWidget(self.view_ylabels, 1, 0)
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

        label = QLabel("Digits")
        self.btn_layout.addWidget(label)
        spin = QSpinBox(self)
        spin.valueChanged.connect(self.digits_changed)
        self.digits_spinbox = spin
        self.btn_layout.addWidget(spin)

        scientific = QCheckBox(_('Scientific'))
        scientific.stateChanged.connect(self.scientific_changed)
        self.scientific_checkbox = scientific
        self.btn_layout.addWidget(scientific)

        gradient_chooser = QComboBox()
        gradient_chooser.setMaximumSize(120, 20)
        gradient_chooser.setIconSize(QSize(100, 20))

        pixmap = QPixmap(100, 15)
        pixmap.fill(Qt.white)
        gradient_chooser.addItem(QIcon(pixmap), " ")

        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        for name, gradient in available_gradients:
            qgradient = gradient.as_qgradient()

            # * fill with white because gradient can be transparent and if we do not "start from whilte", it skews the
            #   colors.
            # * 1 and 13 instead of 0 and 15 to have a transparent border around/between the gradients
            painter.fillRect(0, 1, 100, 13, Qt.white)
            painter.fillRect(0, 1, 100, 13, qgradient)
            gradient_chooser.addItem(QIcon(pixmap), name, gradient)

        # without this, we can crash python :)
        del painter, pixmap
        # select default gradient
        gradient_chooser.setCurrentText(bg_gradient)
        gradient_chooser.currentIndexChanged.connect(self.gradient_changed)
        self.btn_layout.addWidget(gradient_chooser)
        self.gradient_chooser = gradient_chooser

        # Set widget layout
        layout = QVBoxLayout()
        layout.addLayout(self.filters_layout)
        layout.addWidget(array_frame)
        layout.addLayout(self.btn_layout)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.set_data(data, bg_value=bg_value)
        self.model_data.set_bg_gradient(gradient_map[bg_gradient])

        # See http://doc.qt.io/qt-4.8/qt-draganddrop-fridgemagnets-dragwidget-cpp.html for an example
        self.setAcceptDrops(True)

    def gradient_changed(self, index):
        gradient = self.gradient_chooser.itemData(index) if index > 0 else None
        self.model_data.set_bg_gradient(gradient)

    def mousePressEvent(self, event):
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
                previous_index, success = event.mimeData().data("application/x-axis-index").toInt()
                new_index = self.filters_layout.indexOf(self.childAt(event.pos())) // 2

                la_data = self.data_adapter.get_data()
                new_axes = la_data.axes.copy()
                new_axes.insert(new_index, new_axes.pop(new_axes[previous_index]))
                la_data = la_data.transpose(new_axes)
                bg_value = self.data_adapter.bg_value
                if bg_value is not None:
                    bg_value = bg_value.transpose(new_axes)
                self.set_data(la_data, bg_value)

                event.setDropAction(Qt.MoveAction)
                event.accept()
            else:
                event.acceptProposedAction()
        else:
            event.ignore()

    def set_data(self, data=None, bg_value=None):
        # update adapter
        self.data_adapter.set_data(data, bg_value=bg_value)
        la_data = self.data_adapter.get_data()
        axes = la_data.axes
        display_names = axes.display_names

        # update data format and bgcolor
        self._update_digits_scientific(la_data)

        # update filters
        filters_layout = self.filters_layout
        clear_layout(filters_layout)
        # data.size > 0 to avoid arrays with length 0 axes and len(axes) > 0 to avoid scalars (scalar.size == 1)
        if la_data.size > 0 and len(axes) > 0:
            filters_layout.addWidget(QLabel(_("Filters")))
            for axis, display_name in zip(axes, display_names):
                filters_layout.addWidget(QLabel(display_name))
                # FIXME: on very large axes, this is getting too slow. Ideally the combobox should use a model which
                # only fetch labels when they are needed to be displayed
                if len(axis) < 10000:
                    filters_layout.addWidget(self.create_filter_combo(axis))
                else:
                    filters_layout.addWidget(QLabel("too big to be filtered"))
            filters_layout.addStretch()
        self.data_adapter.update_filtered_data({})

        # reset default size
        self.view_axes.set_default_size()
        self.view_ylabels.set_default_size()
        self.view_xlabels.set_default_size()
        self.view_data.set_default_size()

    def _update_digits_scientific(self, data):
        """
        data : LArray
        """
        # TODO: Adapter must provide a method to return a data sample as a Numpy array
        assert isinstance(data, la.LArray)
        data = data.data
        size, dtype = data.size, data.dtype
        # this will yield a data sample of max 199
        step = (size // 100) if size > 100 else 1
        data_sample = data.flat[::step]

        # TODO: refactor so that the expensive format_helper is not called
        # twice (or the values are cached)
        use_scientific = self.choose_scientific(data_sample)

        # XXX: self.ndecimals vs self.digits
        self.digits = self.choose_ndecimals(data_sample, use_scientific)
        self.use_scientific = use_scientific
        self.model_data.set_format(self.cell_format)

        self.digits_spinbox.setValue(self.digits)
        self.digits_spinbox.setEnabled(is_number(dtype))

        self.scientific_checkbox.setChecked(use_scientific)
        self.scientific_checkbox.setEnabled(is_number(dtype))

        self.gradient_chooser.setEnabled(self.model_data.bgcolor_possible)

    def choose_scientific(self, data):
        # max_digits = self.get_max_digits()
        # default width can fit 8 chars
        # FIXME: use max_digits?
        avail_digits = 8
        if data.dtype.type in (np.str, np.str_, np.bool_, np.bool, np.object_):
            return False

        frac_zeros, int_digits, _ = self.format_helper(data)

        # if there are more integer digits than we can display or we can
        # display more information by using scientific format, do so
        # (scientific format "uses" 4 digits, so we win if have >= 4 zeros
        #  -- *including the integer one*)
        # TODO: only do so if we would actually display more information
        # 0.00001 can be displayed with 8 chars
        # 1e-05
        # would
        return int_digits > avail_digits or frac_zeros >= 4

    def choose_ndecimals(self, data, scientific):
        if data.dtype.type in (np.str, np.str_, np.bool_, np.bool, np.object_):
            return 0

        # max_digits = self.get_max_digits()
        # default width can fit 8 chars
        # FIXME: use max_digits?
        avail_digits = 8
        data_frac_digits = self._data_digits(data)
        _, int_digits, negative = self.format_helper(data)
        if scientific:
            int_digits = 2 if negative else 1
            exp_digits = 4
        else:
            exp_digits = 0
        # - 1 for the dot
        ndecimals = avail_digits - 1 - int_digits - exp_digits

        if ndecimals < 0:
            ndecimals = 0

        if data_frac_digits < ndecimals:
            ndecimals = data_frac_digits
        return ndecimals

    def format_helper(self, data):
        if not data.size:
            return 0, 0, False
        data = np.where(np.isfinite(data), data, 0)
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        absmax = max(abs(vmin), abs(vmax))
        logabsmax = math.log10(absmax) if absmax else 0
        # minimum number of zeros before meaningful fractional part
        frac_zeros = math.ceil(-logabsmax) - 1 if logabsmax < 0 else 0
        int_digits = max(ndigits(vmin), ndigits(vmax))
        return frac_zeros, int_digits, vmin < 0

    def get_max_digits(self, need_sign=False, need_dot=False, scientific=False):
        font = get_font("arreditor")  # QApplication.font()
        col_width = 60
        margin_width = 6  # a wild guess
        avail_width = col_width - margin_width
        metrics = QFontMetrics(font)

        def str_width(c):
            return metrics.size(Qt.TextSingleLine, c).width()

        digit_width = max(str_width(str(i)) for i in range(10))
        dot_width = str_width('.')
        sign_width = max(str_width('+'), str_width('-'))
        if need_sign:
            avail_width -= sign_width
        if need_dot:
            avail_width -= dot_width
        if scientific:
            avail_width -= str_width('e') + sign_width + 2 * digit_width
        return avail_width // digit_width

    def _data_digits(self, data, maxdigits=6):
        if not data.size:
            return 0
        threshold = 10 ** -(maxdigits + 1)
        for ndigits in range(maxdigits):
            maxdiff = np.max(np.abs(data - np.round(data, ndigits)))
            if maxdiff < threshold:
                return ndigits
        return maxdigits

    def autofit_columns(self):
        self.view_axes.autofit_columns()
        for column in range(self.model_axes.columnCount()):
            self.resize_axes_column_to_contents(column)
        self.view_xlabels.autofit_columns()
        for column in range(self.model_xlabels.columnCount()):
            self.resize_xlabels_column_to_contents(column)

    def resize_axes_column_to_contents(self, column):
        # must be connected to view_axes.horizontalHeader().sectionHandleDoubleClicked signal
        width = max(self.view_axes.horizontalHeader().sectionSize(column),
                    self.view_ylabels.sizeHintForColumn(column))
        # no need to call resizeSection on view_ylabels (see synchronization lines in init)
        self.view_axes.horizontalHeader().resizeSection(column, width)

    def resize_xlabels_column_to_contents(self, column):
        # must be connected to view_labels.horizontalHeader().sectionHandleDoubleClicked signal
        width = max(self.view_xlabels.horizontalHeader().sectionSize(column),
                    self.view_data.sizeHintForColumn(column))
        # no need to call resizeSection on view_data (see synchronization lines in init)
        self.view_xlabels.horizontalHeader().resizeSection(column, width)

    def resize_axes_row_to_contents(self, row):
        # must be connected to view_axes.verticalHeader().sectionHandleDoubleClicked
        height = max(self.view_axes.verticalHeader().sectionSize(row),
                     self.view_xlabels.sizeHintForRow(row))
        # no need to call resizeSection on view_xlabels (see synchronization lines in init)
        self.view_axes.verticalHeader().resizeSection(row, height)

    def resize_ylabels_row_to_contents(self, row):
        # must be connected to view_labels.verticalHeader().sectionHandleDoubleClicked
        height = max(self.view_ylabels.verticalHeader().sectionSize(row),
                     self.view_data.sizeHintForRow(row))
        # no need to call resizeSection on view_data (see synchronization lines in init)
        self.view_ylabels.verticalHeader().resizeSection(row, height)

    @property
    def dirty(self):
        self.data_adapter.update_changes()
        return len(self.data_adapter.changes) > 0

    def accept_changes(self):
        """Accept changes"""
        la_data = self.data_adapter.accept_changes()
        self._update_digits_scientific(la_data)

    def reject_changes(self):
        """Reject changes"""
        self.data_adapter.reject_changes()

    @property
    def cell_format(self):
        type = self.data_adapter.dtype.type
        if type in (np.str, np.str_, np.bool_, np.bool, np.object_):
            return '%s'
        else:
            format_letter = 'e' if self.use_scientific else 'f'
            return '%%.%d%s' % (self.digits, format_letter)

    def scientific_changed(self, value):
        self.use_scientific = value
        self.digits = self.choose_ndecimals(self.data_adapter.get_data(), value)
        self.digits_spinbox.setValue(self.digits)
        self.model_data.set_format(self.cell_format)

    def digits_changed(self, value):
        self.digits = value
        self.model_data.set_format(self.cell_format)

    def create_filter_combo(self, axis):
        def filter_changed(checked_items):
            self.data_adapter.change_filter(axis, checked_items)
        combo = FilterComboBox(self)
        combo.addItems([str(l) for l in axis.labels])
        combo.checkedItemsChanged.connect(filter_changed)
        return combo

    def _selection_data(self, headers=True, none_selects_all=True):
        """
        Returns an iterator over selected labels and data
        if headers=True and a Numpy ndarray containing only
        the data otherwise.

        Parameters
        ----------
        headers : bool, optional
            Labels are also returned if True.
        none_selects_all : bool, optional
            If True (default) and selection is empty, returns all data.

        Returns
        -------
        numpy.ndarray or itertools.chain
        """
        bounds = self.view_data._selection_bounds(none_selects_all=none_selects_all)
        if bounds is None:
            return None
        row_min, row_max, col_min, col_max = bounds
        raw_data = self.model_data.get_values(row_min, col_min, row_max, col_max)
        if headers:
            if not self.data_adapter.ndim:
                return raw_data
            # FIXME: this is extremely ad-hoc.
            # TODO: in the future (pandas-based branch) we should use to_string(data[self._selection_filter()])
            dim_headers = self.model_axes.get_values()
            xlabels = self.model_xlabels.get_values(top=col_min, bottom=col_max)
            topheaders = [[dim_header[0] for dim_header in dim_headers] + [label[0] for label in xlabels]]
            if self.data_adapter.ndim == 1:
                return chain(topheaders, [chain([''], row) for row in raw_data])
            else:
                assert self.data_adapter.ndim > 1
                ylabels = self.model_ylabels.get_values(left=row_min, right=row_max)
                return chain(topheaders,
                             [chain([ylabels[j][r] for j in range(len(ylabels))], row)
                              for r, row in enumerate(raw_data)])
        else:
            return raw_data

    def copy(self):
        """Copy selection as text to clipboard"""
        data = self._selection_data()
        if data is None:
            return

        # np.savetxt make things more complicated, especially on py3
        # XXX: why don't we use repr for everything?
        def vrepr(v):
            if isinstance(v, float):
                return repr(v)
            else:
                return str(v)
        text = '\n'.join('\t'.join(vrepr(v) for v in line) for line in data)
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    def to_excel(self):
        """View selection in Excel"""
        if xw is None:
            QMessageBox.critical(self, "Error", "to_excel() is not available because xlwings is not installed")
        data = self._selection_data()
        if data is None:
            return
        # convert (row) generators to lists then array
        # TODO: the conversion to array is currently necessary even though xlwings will translate it back to a list
        #       anyway. The problem is that our lists contains numpy types and especially np.str_ crashes xlwings.
        #       unsure how we should fix this properly: in xlwings, or change _selection_data to return only standard
        #       Python types.
        xw.view(np.array([list(r) for r in data]))

    def paste(self):
        bounds = self.view_data._selection_bounds()
        if bounds is None:
            return
        row_min, row_max, col_min, col_max = bounds
        clipboard = QApplication.clipboard()
        text = str(clipboard.text())
        list_data = [line.split('\t') for line in text.splitlines()]
        try:
            # take the first cell which contains '\'
            pos_last = next(i for i, v in enumerate(list_data[0]) if '\\' in v)
        except StopIteration:
            # if there isn't any, assume 1d array
            pos_last = 0
        if pos_last:
            # ndim > 1
            list_data = [line[pos_last + 1:] for line in list_data[1:]]
        elif len(list_data) == 2 and list_data[1][0] == '':
            # ndim == 1
            list_data = [list_data[1][1:]]
        new_data = np.array(list_data)
        if new_data.shape[0] > 1:
            row_max = row_min + new_data.shape[0]
        if new_data.shape[1] > 1:
            col_max = col_min + new_data.shape[1]

        result = self.model_data.set_values(row_min, col_min, row_max, col_max, new_data)
        if result is None:
            return

        # TODO: when pasting near bottom/right boundaries and size of
        # new_data exceeds destination size, we should either have an error
        # or clip new_data
        self.view_data.selectionModel().select(QItemSelection(*result), QItemSelectionModel.ClearAndSelect)

    def plot(self):
        from matplotlib.figure import Figure
        from larray_editor.utils import show_figure

        data = self._selection_data(headers=False)
        if data is None:
            return

        row_min, row_max, col_min, col_max = self.view_data._selection_bounds()
        dim_names = self.data_adapter.get_axes_names()
        # labels
        xlabels = [label[0] for label in self.model_xlabels.get_values(top=col_min, bottom=col_max)]
        ylabels = self.model_ylabels.get_values(left=row_min, right=row_max)
        # transpose ylabels
        ylabels = [[str(ylabels[i][j]) for i in range(len(ylabels))] for j in range(len(ylabels[0]))]
        # if there is only one dimension, ylabels is empty
        if not ylabels:
            ylabels = [[]]

        assert data.ndim == 2

        figure = Figure()

        # create an axis
        ax = figure.add_subplot(111)

        if data.shape[1] == 1:
            # plot one column
            xlabel = ','.join(dim_names[:-1])
            xticklabels = ['\n'.join(row) for row in ylabels]
            xdata = np.arange(row_max - row_min)
            ax.plot(xdata, data[:, 0])
            ax.set_ylabel(xlabels[0])
        else:
            # plot each row as a line
            xlabel = dim_names[-1]
            xticklabels = [str(label) for label in xlabels]
            xdata = np.arange(col_max - col_min)
            for row in range(len(data)):
                ax.plot(xdata, data[row], label=' '.join(ylabels[row]))

        # set x axis
        ax.set_xlabel(xlabel)
        ax.set_xlim((xdata[0], xdata[-1]))
        # we need to do that because matplotlib is smart enough to
        # not show all ticks but a selection. However, that selection
        # may include ticks outside the range of x axis
        xticks = [t for t in ax.get_xticks().astype(int) if t <= len(xticklabels) - 1]
        xticklabels = [xticklabels[t] for t in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        if data.shape[1] != 1 and ylabels != [[]]:
            # set legend
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.legend()

        show_figure(self, figure)
