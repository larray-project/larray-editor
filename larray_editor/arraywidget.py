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

import math
import logging

import numpy as np

from qtpy.QtCore import Qt, QPoint, QItemSelection, QItemSelectionModel, Signal, QSize
from qtpy.QtGui import QDoubleValidator, QIntValidator, QKeySequence, QFontMetrics, QCursor, QPixmap, QPainter, QIcon
from qtpy.QtWidgets import (QApplication, QTableView, QItemDelegate, QLineEdit, QCheckBox,
                            QMessageBox, QMenu, QLabel, QSpinBox, QWidget, QToolTip, QShortcut, QScrollBar,
                            QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy, QFrame, QComboBox)

from larray_editor.utils import (keybinding, create_action, clear_layout, get_default_font, is_number, is_float, _,
                                 ima, LinearGradient, logger, cached_property)
from larray_editor.arrayadapter import get_adapter
from larray_editor.arraymodel import LabelsArrayModel, AxesArrayModel, DataArrayModel
from larray_editor.combo import FilterComboBox


# XXX: define Enum instead ?
TOP, BOTTOM = 0, 1
LEFT, RIGHT = 0, 1


class AbstractView(QTableView):
    """Abstract view class"""
    def __init__(self, parent, model, hpos, vpos):
        QTableView.__init__(self, parent)

        # set model
        self.setModel(model)

        # set position
        if not (hpos == LEFT or hpos == RIGHT):
            raise TypeError(f"Value of hpos must be {LEFT} or {RIGHT}")
        self.hpos = hpos
        if not (vpos == TOP or vpos == BOTTOM):
            raise TypeError(f"Value of vpos must be {TOP} or {BOTTOM}")
        self.vpos = vpos

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

    def set_default_size(self):
        # make the grid a bit more compact
        self.horizontalHeader().setDefaultSectionSize(64)
        self.verticalHeader().setDefaultSectionSize(20)
        if self.vpos == TOP:
            self.horizontalHeader().setFixedHeight(10)
        if self.hpos == LEFT:
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
    def __init__(self, dtype, parent=None, font=None, minvalue=None, maxvalue=None):
        QItemDelegate.__init__(self, parent)
        self.dtype = dtype
        if font is None:
            font = get_default_font()
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
        # TODO: dtype should be taken from the adapter instead. Only the adapter knows whether the dtype is per cell
        #  (e.g. list), per column (e.g. Dataframe) or homogenous for the whole table (e.g. la.Array)
        # dtype = model.adapter.get_dtype(hpos, vpos)
        dtype = self.dtype
        value = model.get_value(index)
        if dtype.name == "bool":
            # toggle value
            value = not value
            model.setData(index, value)
            return
        elif value is not np.ma.masked:
            # Not using a QSpinBox for integer inputs because I could not find
            # a way to prevent the spinbox/editor from closing if the value is
            # invalid. Using the builtin minimum/maximum of the spinbox works
            # but that provides no message so it is less clear.
            editor = QLineEdit(parent)
            if is_number(dtype):
                minvalue, maxvalue = self.minvalue, self.maxvalue
                validator = QDoubleValidator(editor) if is_float(dtype) else QIntValidator(editor)
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

    def set_dtype(self, dtype):
        model = self.model()
        delegate = ArrayDelegate(dtype, self, minvalue=model.minvalue, maxvalue=model.maxvalue)
        self.setItemDelegate(delegate)

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

    def contextMenuEvent(self, event):
        """Reimplement Qt method"""
        self.context_menu.popup(event.globalPos())
        event.accept()

    def keyPressEvent(self, event):
        """Reimplement Qt method"""

        # allow to start editing cells by pressing Enter
        if event.key() == Qt.Key_Return and not self.model().readonly:
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
    def __init__(self, parent, data_scrollbar):
        super().__init__(data_scrollbar.orientation(), parent)
        self.setMinimum(data_scrollbar.minimum())
        self.setMaximum(data_scrollbar.maximum())
        self.setSingleStep(data_scrollbar.singleStep())
        self.setPageStep(data_scrollbar.pageStep())

        data_scrollbar.valueChanged.connect(self.setValue)
        self.valueChanged.connect(data_scrollbar.setValue)

        data_scrollbar.rangeChanged.connect(self.setRange)
        self.rangeChanged.connect(data_scrollbar.setRange)


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
        self._used_font = data_model.font

    def font_changed(self):
        model_font = self.data_model.font
        changed = model_font is not self._used_font and model_font != self._used_font
        if changed:
            self._used_font = model_font
        return changed

    @cached_property(font_changed)
    def str_width(self):
        # font_metrics = QFontMetrics(self._used_font)
        # def str_width(c):
        #     return font_metrics.size(Qt.TextSingleLine, c).width()
        # return str_width
        return QFontMetrics(self._used_font).width

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
        self.model_axes = AxesArrayModel(parent=self, readonly=readonly)
        self.view_axes = AxesView(parent=self, model=self.model_axes)

        self.model_hlabels = LabelsArrayModel(parent=self, readonly=readonly)
        self.view_hlabels = LabelsView(parent=self, model=self.model_hlabels, hpos=RIGHT, vpos=TOP)

        self.model_vlabels = LabelsArrayModel(parent=self, readonly=readonly)
        self.view_vlabels = LabelsView(parent=self, model=self.model_vlabels, hpos=LEFT, vpos=BOTTOM)

        self.model_data = DataArrayModel(parent=self, readonly=readonly, minvalue=minvalue, maxvalue=maxvalue)
        self.view_data = DataView(parent=self, model=self.model_data)

        self.font_metrics = FontMetrics(self.model_data)

        # in case data is None
        self.data_adapter = None

        # Create vertical and horizontal scrollbars
        self.vscrollbar = ScrollBar(self, self.view_data.verticalScrollBar())
        self.hscrollbar = ScrollBar(self, self.view_data.horizontalScrollBar())

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

        # Synchronize scrolling
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
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # set gradient
        self.model_data.set_bg_gradient(gradient_map[bg_gradient])

        # set data
        if data is not None:
            self.set_data(data, bg_value=bg_value, frac_digits=digits)

        # See http://doc.qt.io/qt-4.8/qt-draganddrop-fridgemagnets-dragwidget-cpp.html for an example
        self.setAcceptDrops(True)

    def gradient_changed(self, index):
        gradient = self.gradient_chooser.itemData(index) if index > 0 else None
        self.model_data.set_bg_gradient(gradient)

    def data_changed(self, data_model_changes):
        changes = self.data_adapter.translate_changes(data_model_changes)
        self.dataChanged.emit(changes)

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

    def _update_models(self, reset_model_data, reset_minmax):
        # axes names
        axes_names = self.data_adapter.get_axes_names(fold_last_axis=True)
        self.model_axes.set_data(axes_names)
        # horizontal labels
        hlabels = self.data_adapter.get_hlabels()
        self.model_hlabels.set_data(hlabels)
        # vertical labels
        vlabels = self.data_adapter.get_vlabels()
        self.model_vlabels.set_data(vlabels)
        # raw data
        # use flag reset=False to avoid calling reset() several times
        raw_data = self.data_adapter.get_raw_data()
        self.model_data.set_data(raw_data, reset=False)
        # bg value
        # use flag reset=False to avoid calling reset() several times
        bg_value = self.data_adapter.get_bg_value()
        self.model_data.set_bg_value(bg_value, reset=False)
        # reset min and max values if required
        if reset_minmax:
            self.model_data.reset_minmax()
        # reset the data model if required
        if reset_model_data:
            self.model_data.reset()

    def set_data(self, data, bg_value=None, frac_digits=None):
        # get new adapter instance + set data
        self.data_adapter = get_adapter(data=data, bg_value=bg_value)
        # update filters
        self._update_filter()
        # update models
        # Note: model_data is reset by call of set_format below
        self._update_models(reset_model_data=False, reset_minmax=True)
        # reset default size
        self._reset_default_size()
        # update data format
        self.set_format(frac_digits=frac_digits, scientific=None)
        # update gradient_chooser
        self.gradient_chooser.setEnabled(self.model_data.bgcolor_possible)
        # update dtype in view_data
        self.view_data.set_dtype(self.data_adapter.dtype)

    def _reset_default_size(self):
        self.view_axes.set_default_size()
        self.view_vlabels.set_default_size()
        self.view_hlabels.set_default_size()
        self.view_data.set_default_size()

    def _update_filter(self):
        filters_layout = self.filters_layout
        clear_layout(filters_layout)
        axes = self.data_adapter.get_axes_filtered_data()
        # size > 0 to avoid arrays with length 0 axes and len(axes) > 0 to avoid scalars (scalar.size == 1)
        if self.data_adapter.size > 0 and len(axes) > 0:
            filters_layout.addWidget(QLabel(_("Filters")))
            for axis in axes:
                filters_layout.addWidget(QLabel(axis.name))
                # FIXME: on very large axes, this is getting too slow. Ideally the combobox should use a model which
                # only fetch labels when they are needed to be displayed
                if len(axis) < 10000:
                    filters_layout.addWidget(self.create_filter_combo(axis))
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

        data_sample = self.data_adapter.get_finite_sample()
        is_number_dtype = np.issubdtype(data_sample.dtype, np.number)
        cur_colwidth = self._get_current_min_col_width()
        if is_number_dtype and data_sample.size:
            # TODO: this should come from the adapter or from the data_model (were it is already computed!!!)
            #       (but modified whenever the data changes)
            vmin, vmax = np.min(data_sample), np.max(data_sample)
            int_digits = max(num_int_digits(vmin), num_int_digits(vmax))
            has_negative = vmin < 0

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
                absmax = max(abs(vmin), abs(vmax))
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
                frac_digits = self._data_frac_digits(data_sample, max_frac_digits=max_frac_digits)
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
            data_colwidth = 60

        self.model_data.set_format(fmt, reset=True)

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

        if not scientific_toggled or data_colwidth > cur_colwidth:
            header = self.view_hlabels.horizontalHeader()

            # FIXME: this will set width of the 40 first columns (otherwise it gets very slow, eg. big1d)
            #        but I am not eager to fix this before merging the buffer branch
            num_cols = min(header.count(), 40)
            hlabels = self.model_hlabels.get_values(bottom=num_cols)
            str_width = FontMetrics(self.model_hlabels).str_width

            MIN_COLWITH = 30
            data_colwidth = max(data_colwidth, MIN_COLWITH)

            MARGIN_WIDTH = 8  # empirically measured

            def get_header_width(i):
                return MARGIN_WIDTH + max(str_width(str(label)) for label in hlabels[i])

            for i in range(num_cols):
                colwidth = max(get_header_width(i), data_colwidth)
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

    def change_filter(self, axis, indices):
        self.data_adapter.update_filter(axis, indices)
        self._update_models(reset_model_data=True, reset_minmax=False)

    def create_filter_combo(self, axis):
        def filter_changed(checked_items):
            self.change_filter(axis, checked_items)
        combo = FilterComboBox(self)
        combo.addItems([str(label) for label in axis.labels])
        combo.checkedItemsChanged.connect(filter_changed)
        return combo

    def _selection_data(self, headers=True, none_selects_all=True):
        """
        Return selected labels as lists and raw data as Numpy ndarray
        if headers=True or only the raw data otherwise

        Parameters
        ----------
        headers : bool, optional
            Labels are also returned if True.
        none_selects_all : bool, optional
            If True (default) and selection is empty, returns all data.

        Returns
        -------
        raw_data: numpy.ndarray
        axes_names: list
        vlabels: nested list
        hlabels: list
        """
        bounds = self.view_data._selection_bounds(none_selects_all=none_selects_all)
        if bounds is None:
            return None
        row_min, row_max, col_min, col_max = bounds
        raw_data = self.model_data.get_values(row_min, col_min, row_max, col_max)
        if headers:
            # FIXME: using data_adapter.ndim here and in the vlabels line below is
            #        inherently buggy, because this does not take filter into account,
            #        which should be the case for selection-related stuff which work
            #        on visible data
            if not self.data_adapter.ndim:
                return raw_data, None, None, None
            axes_names = self.model_axes.get_values()
            if len(axes_names):
                hlabels = [label[0]
                           for label in self.model_hlabels.get_values(top=col_min, bottom=col_max)]
            else:
                hlabels = []
            vlabels = self.model_vlabels.get_values(left=row_min, right=row_max) if self.data_adapter.ndim > 1 else []
            return raw_data, axes_names, vlabels, hlabels
        else:
            return raw_data

    def copy(self):
        """Copy selection as text to clipboard"""
        raw_data, axes_names, vlabels, hlabels = self._selection_data()
        data = self.data_adapter.selection_to_chain(raw_data, axes_names, vlabels, hlabels)
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
        """Export selection in Excel"""
        raw_data, axes_names, vlabels, hlabels = self._selection_data()
        try:
            self.data_adapter.to_excel(raw_data, axes_names, vlabels, hlabels)
        except ImportError:
            QMessageBox.critical(self, "Error", "to_excel() is not available because xlwings is not installed")

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
        if pos_last or '\\' in list_data[0][0]:
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
        from larray_editor.utils import show_figure
        from larray_editor.editor import AbstractEditor, MappingEditor
        raw_data, axes_names, vlabels, hlabels = self._selection_data()
        try:
            figure = self.data_adapter.plot(raw_data, axes_names, vlabels, hlabels)
            widget = self
            while widget is not None and not isinstance(widget, AbstractEditor) and callable(widget.parent):
                widget = widget.parent()
            title = widget.current_array_name if isinstance(widget, MappingEditor) else None
            show_figure(self, figure, title)
        except ImportError:
            QMessageBox.critical(self, "Error", "plot() is not available because matplotlib is not installed")
