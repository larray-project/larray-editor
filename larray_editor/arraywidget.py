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
from qtpy.QtCore import Qt, QPoint, QItemSelection, QItemSelectionModel, QItemSelectionRange, Slot
from qtpy.QtGui import QDoubleValidator, QIntValidator, QKeySequence, QFontMetrics, QCursor
from qtpy.QtWidgets import (QApplication, QHBoxLayout, QTableView, QItemDelegate, QLineEdit, QCheckBox,
                            QMessageBox, QMenu, QLabel, QSpinBox, QWidget, QVBoxLayout, QToolTip, QShortcut)

try:
    import xlwings as xw
except ImportError:
    xw = None

from larray_editor.utils import (keybinding, create_action, clear_layout, get_font, from_qvariant, to_qvariant,
                                 is_number, is_float, _, ima)
from larray_editor.arraymodel import ArrayModel
from larray_editor.combo import FilterComboBox, FilterMenu
import larray as la


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


class ArrayView(QTableView):
    """Array view class"""
    def __init__(self, parent, model, dtype, shape):
        QTableView.__init__(self, parent)

        self.setModel(model)
        delegate = ArrayDelegate(dtype, self,
                                 minvalue=model.minvalue,
                                 maxvalue=model.maxvalue)
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
            (keybinding('Copy'), self.copy),
            (QKeySequence("Ctrl+E"), self.to_excel),
            (keybinding('Paste'), self.paste),
            (keybinding('Print'), self.plot)
        ]
        for key_seq, target in shortcuts:
            shortcut = QShortcut(key_seq, self)
            shortcut.activated.connect(target)

        # make the grid a bit more compact
        self.horizontalHeader().setDefaultSectionSize(64)
        self.verticalHeader().setDefaultSectionSize(20)

        self.horizontalScrollBar().valueChanged.connect(
            self.on_horizontal_scroll_changed)
        self.verticalScrollBar().valueChanged.connect(
            self.on_vertical_scroll_changed)
        # self.horizontalHeader().sectionClicked.connect(
        #     self.on_horizontal_header_clicked)

    def on_horizontal_header_clicked(self, section_index):
        menu = FilterMenu(self)
        header = self.horizontalHeader()
        headerpos = self.mapToGlobal(header.pos())
        posx = headerpos.x() + header.sectionPosition(section_index)
        posy = headerpos.y() + header.height()
        menu.exec_(QPoint(posx, posy))

    def on_vertical_scroll_changed(self, value):
        if value == self.verticalScrollBar().maximum():
            self.model().fetch_more_rows()

    def on_horizontal_scroll_changed(self, value):
        if value == self.horizontalScrollBar().maximum():
            self.model().fetch_more_columns()

    def setup_context_menu(self):
        """Setup context menu"""
        self.copy_action = create_action(self, _('Copy'),
                                         shortcut=keybinding('Copy'),
                                         icon=ima.icon('edit-copy'),
                                         triggered=self.copy)
        self.excel_action = create_action(self, _('Copy to Excel'),
                                          shortcut="Ctrl+E",
                                          # icon=ima.icon('edit-copy'),
                                          triggered=self.to_excel)
        self.paste_action = create_action(self, _('Paste'),
                                          shortcut=keybinding('Paste'),
                                          icon=ima.icon('edit-paste'),
                                          triggered=self.paste)
        self.plot_action = create_action(self, _('Plot'),
                                         shortcut=keybinding('Print'),
                                         # icon=ima.icon('editcopy'),
                                         triggered=self.plot)
        menu = QMenu(self)
        menu.addActions([self.copy_action, self.excel_action, self.plot_action,
                         self.paste_action])
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
            self.plot()
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
        assert len(selection) == 1
        srange = selection[0]
        assert isinstance(srange, QItemSelectionRange)
        xoffset = len(self.model().xlabels) - 1
        yoffset = len(self.model().ylabels) - 1
        row_min = max(srange.top() - xoffset, 0)
        row_max = max(srange.bottom() - xoffset, 0)
        col_min = max(srange.left() - yoffset, 0)
        col_max = max(srange.right() - yoffset, 0)
        # if not all rows/columns have been loaded
        if row_min == 0 and row_max == self.model().rows_loaded - 1:
            row_max = self.model().total_rows - 1
        if col_min == 0 and col_max == self.model().cols_loaded - 1:
            col_max = self.model().total_cols - 1
        return row_min, row_max + 1, col_min, col_max + 1

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
        bounds = self._selection_bounds(none_selects_all=none_selects_all)
        if bounds is None:
            return None
        row_min, row_max, col_min, col_max = bounds
        raw_data = self.model().get_values(row_min, col_min, row_max, col_max)
        if headers:
            xlabels = self.model().xlabels
            ylabels = self.model().ylabels
            # FIXME: this is extremely ad-hoc. We should either use
            # model.data.ndim (orig_ndim?) or add a new concept (eg dim_names)
            # in addition to xlabels & ylabels,
            # TODO: in the future (pandas-based branch) we should use
            # to_string(data[self._selection_filter()])
            dim_names = xlabels[0]
            if len(dim_names) > 1:
                dim_headers = dim_names[:-2] + [dim_names[-2] + ' \\ ' +
                                                dim_names[-1]]
            else:
                dim_headers = dim_names
            topheaders = [dim_headers + list(xlabels[i][col_min:col_max])
                          for i in range(1, len(xlabels))]
            if not dim_names:
                return raw_data
            elif len(dim_names) == 1:
                # 1 dimension
                return chain(topheaders, [chain([''], row) for row in raw_data])
            else:
                # >1 dimension
                assert len(dim_names) > 1
                return chain(topheaders,
                             [chain([ylabels[j][r + row_min]
                                     for j in range(1, len(ylabels))],
                                    row)
                              for r, row in enumerate(raw_data)])
        else:
            return raw_data

    @Slot()
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

    @Slot()
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

    @Slot()
    def paste(self):
        model = self.model()
        bounds = self._selection_bounds()
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

        result = model.set_values(row_min, col_min, row_max, col_max, new_data)
        if result is None:
            return

        # TODO: when pasting near bottom/right boundaries and size of
        # new_data exceeds destination size, we should either have an error
        # or clip new_data
        self.selectionModel().select(QItemSelection(*result),
                                     QItemSelectionModel.ClearAndSelect)

    def plot(self):
        from matplotlib.figure import Figure
        from larray_editor.utils import show_figure

        data = self._selection_data(headers=False)
        if data is None:
            return

        row_min, row_max, col_min, col_max = self._selection_bounds()
        dim_names = self.model().xlabels[0]
        # label for each selected column
        xlabels = self.model().xlabels[1][col_min:col_max]
        # list of selected labels for each index column
        labels_per_index_column = [col_labels[row_min:row_max] for col_labels in self.model().ylabels[1:]]
        # list of (str) label for each selected row
        ylabels = [[str(label) for label in row_labels]
                   for row_labels in zip(*labels_per_index_column)]
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
            xticklabels = ['\n'.join(ylabels[row]) for row in range(row_max - row_min)]
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


class ArrayEditorWidget(QWidget):
    def __init__(self, parent, data, readonly=False, bg_value=None, bg_gradient=None, minvalue=None, maxvalue=None):
        QWidget.__init__(self, parent)
        readonly = np.isscalar(data)
        self.model = ArrayModel(data, readonly=readonly, parent=self,
                                bg_value=bg_value, bg_gradient=bg_gradient,
                                minvalue=minvalue, maxvalue=maxvalue)
        self.view = ArrayView(self, self.model, data.dtype, data.shape)

        self.filters_layout = QHBoxLayout()
        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignLeft)

        label = QLabel("Digits")
        btn_layout.addWidget(label)
        spin = QSpinBox(self)
        spin.valueChanged.connect(self.digits_changed)
        self.digits_spinbox = spin
        btn_layout.addWidget(spin)

        scientific = QCheckBox(_('Scientific'))
        scientific.stateChanged.connect(self.scientific_changed)
        self.scientific_checkbox = scientific
        btn_layout.addWidget(scientific)

        bgcolor = QCheckBox(_('Background color'))
        bgcolor.stateChanged.connect(self.model.bgcolor)
        self.bgcolor_checkbox = bgcolor
        btn_layout.addWidget(bgcolor)

        layout = QVBoxLayout()
        layout.addLayout(self.filters_layout)
        layout.addWidget(self.view)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.set_data(data, bg_value=bg_value, bg_gradient=bg_gradient)

        # See http://doc.qt.io/qt-4.8/qt-draganddrop-fridgemagnets-dragwidget-cpp.html for an example
        self.setAcceptDrops(True)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self.dragLabel = self.childAt(event.pos())
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

                la_data = self.model.get_data()
                new_axes = la_data.axes.copy()
                new_axes.insert(new_index, new_axes.pop(new_axes[previous_index]))
                la_data = la_data.transpose(new_axes)
                self.set_data(la_data, self.model.bg_gradient, self.model.bg_value)

                event.setDropAction(Qt.MoveAction)
                event.accept()
            else:
                event.acceptProposedAction()
        else:
            event.ignore()

    def set_data(self, data, bg_gradient=None, bg_value=None):
        la_data = la.aslarray(data)
        axes = la_data.axes
        display_names = axes.display_names

        filters_layout = self.filters_layout
        clear_layout(filters_layout)
        filters_layout.addWidget(QLabel(_("Filters")))
        for axis, display_name in zip(axes, display_names):
            filters_layout.addWidget(QLabel(display_name))
            filters_layout.addWidget(self.create_filter_combo(axis))
        filters_layout.addStretch()

        self.model.set_data(la_data, bg_gradient=bg_gradient, bg_value=bg_value)
        self._update(la_data)

    def _update(self, la_data):
        size = la_data.size
        # this will yield a data sample of max 199
        step = (size // 100) if size > 100 else 1
        data_sample = la_data.data.flat[::step]

        # TODO: refactor so that the expensive format_helper is not called
        # twice (or the values are cached)
        use_scientific = self.choose_scientific(data_sample)

        # XXX: self.ndecimals vs self.digits
        self.digits = self.choose_ndecimals(data_sample, use_scientific)
        self.use_scientific = use_scientific
        self.model.set_format(self.cell_format)

        self.digits_spinbox.setValue(self.digits)
        self.digits_spinbox.setEnabled(is_number(la_data.dtype))

        self.scientific_checkbox.setChecked(use_scientific)
        self.scientific_checkbox.setEnabled(is_number(la_data.dtype))

        self.bgcolor_checkbox.setChecked(self.model.bgcolor_enabled)
        self.bgcolor_checkbox.setEnabled(self.model.bgcolor_enabled)

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

    @property
    def dirty(self):
        self.model.update_global_changes()
        return len(self.model.changes) > 1

    def accept_changes(self):
        """Accept changes"""
        la_data = self.model.accept_changes()
        self._update(la_data)

    def reject_changes(self):
        """Reject changes"""
        self.model.reject_changes()

    @property
    def cell_format(self):
        type = self.model.get_data().dtype.type
        if type in (np.str, np.str_, np.bool_, np.bool, np.object_):
            return '%s'
        else:
            format_letter = 'e' if self.use_scientific else 'f'
            return '%%.%d%s' % (self.digits, format_letter)

    def scientific_changed(self, value):
        self.use_scientific = value
        self.digits = self.choose_ndecimals(self.model.get_data(), value)
        self.digits_spinbox.setValue(self.digits)
        self.model.set_format(self.cell_format)

    def digits_changed(self, value):
        self.digits = value
        self.model.set_format(self.cell_format)

    def create_filter_combo(self, axis):
        def filter_changed(checked_items):
            filtered = self.model.change_filter(axis, checked_items)
            self._update(filtered)
        combo = FilterComboBox(self)
        combo.addItems([str(l) for l in axis.labels])
        combo.checkedItemsChanged.connect(filter_changed)
        return combo
