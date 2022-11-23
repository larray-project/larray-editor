import ast
import warnings

import numpy as np
import larray as la

from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QSplitter, QHBoxLayout,
                            QLabel, QCheckBox, QLineEdit, QComboBox, QMessageBox)

from larray_editor.utils import replace_inf, _
from larray_editor.arraywidget import ArrayEditorWidget
from larray_editor.editor import AbstractEditor, DISPLAY_IN_GRID


class ComparatorWidget(QWidget):
    """Comparator Widget"""
    def __init__(self, parent=None, bg_gradient='red-white-blue', rtol=0, atol=0, nans_equal=True):
        QWidget.__init__(self, parent)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # max diff label
        maxdiff_layout = QHBoxLayout()
        maxdiff_layout.addWidget(QLabel('maximum absolute relative difference:'))
        self.maxdiff_label = QLabel('')
        maxdiff_layout.addWidget(self.maxdiff_label)
        maxdiff_layout.addStretch()
        layout.addLayout(maxdiff_layout)

        self.arraywidget = ArrayEditorWidget(self, data=None, readonly=True, bg_gradient=bg_gradient)

        # show difference only
        diff_checkbox = QCheckBox(_('Differences Only'))
        diff_checkbox.stateChanged.connect(self.display)
        self.diff_checkbox = diff_checkbox
        self.arraywidget.btn_layout.addWidget(diff_checkbox)

        # absolute/relative tolerance
        tolerance_layout = QHBoxLayout()
        tooltip = """Element i of two arrays are considered as equal if they satisfy the following equation:
        abs(array1[i] - array2[i]) <= (absolute_tol + relative_tol * abs(array2[i]))"""

        tolerance_label = QLabel("tolerance:")
        tolerance_label.setToolTip(tooltip)
        self.arraywidget.btn_layout.addWidget(tolerance_label)

        tolerance_combobox = QComboBox()
        tolerance_combobox.addItems(["absolute", "relative"])
        tolerance_combobox.setToolTip(tooltip)
        tolerance_combobox.currentTextChanged.connect(self.update_isequal)
        tolerance_layout.addWidget(tolerance_combobox)
        self.tolerance_combobox = tolerance_combobox

        tolerance_line_edit = QLineEdit()
        tolerance_line_edit.setValidator(QDoubleValidator())
        tolerance_line_edit.setPlaceholderText("1e-8")
        tolerance_line_edit.setMaximumWidth(80)
        tolerance_line_edit.setToolTip("Press Enter to activate the new tolerance value")
        tolerance_line_edit.editingFinished.connect(self.update_isequal)
        tolerance_layout.addWidget(tolerance_line_edit)
        self.tolerance_line_edit = tolerance_line_edit

        self.arraywidget.btn_layout.addLayout(tolerance_layout)

        self.nans_equal = nans_equal

        # add local arraywidget to layout
        self.arraywidget.btn_layout.addStretch()
        layout.addWidget(self.arraywidget)

        self.array = None
        self.array0 = None
        self.isequal = None
        self.bg_value = None
        self.stack_axis = None

        if rtol > 0 and atol > 0:
            raise ValueError("Arguments 'rtol' and 'atol' cannot be used together.")
        if rtol > 0:
            self.tolerance_combobox.setCurrentText("relative")
            self.tolerance_line_edit.setText(str(rtol))
        if atol > 0:
            self.tolerance_combobox.setCurrentText("absolute")
            self.tolerance_line_edit.setText(str(atol))

    # override keyPressEvent to prevent pressing Enter after changing the tolerance value
    # in associated QLineEdit to close the parent dialog box
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            return
        QWidget.keyPressEvent(self, event)

    def set_data(self, arrays, stack_axis):
        """
        Parameters
        ----------
        arrays: list or tuple of scalar, Array, ndarray
            Arrays to compare.
        stack_axis: Axis
            Names of arrays.
        """
        assert all(np.isscalar(a) or isinstance(a, la.Array) for a in arrays)
        self.stack_axis = stack_axis
        try:
            self.array = la.stack(arrays, stack_axis)
            self.array0 = self.array[stack_axis.i[0]]
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.array = la.Array([''])
            self.array0 = self.array
        self.update_isequal()

    def update_isequal(self):
        if self.array is None:
            return

        try:
            tol_str = self.tolerance_line_edit.text()
            tol = ast.literal_eval(tol_str) if tol_str else 0
            atol, rtol = (tol, 0) if self.tolerance_combobox.currentText() == "absolute" else (0, tol)
            self.isequal = self.array.eq(self.array0, rtol=rtol, atol=atol, nans_equal=self.nans_equal)
        except TypeError:
            self.isequal = self.array == self.array0

        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                diff = self.array - self.array0
                reldiff = diff / self.array0
                # this is necessary for nan, inf and -inf (because inf - inf = nan, not 0)
                # this is more precise than divnot0, it only ignore 0 / 0, not x / 0
                reldiff[self.isequal] = 0
                # replace -inf by min(reldiff), +inf by max(reldiff)
                reldiff, relmin, relmax = replace_inf(reldiff)
                maxabsreldiff = max(abs(relmin), abs(relmax))
            if maxabsreldiff:
                # scale reldiff to range 0-1 with 0.5 for reldiff = 0
                self.bg_value = (reldiff / maxabsreldiff) / 2 + 0.5
            else:
                # do NOT use full_like as we don't want to inherit array dtype
                self.bg_value = la.full(self.array.axes, 0.5)
        except TypeError:
            # str/object array
            maxabsreldiff = la.nan
            # do NOT use full_like as we don't want to inherit array dtype
            self.bg_value = la.full(self.array.axes, 0.5)

        self.maxdiff_label.setText(str(maxabsreldiff))
        self.display(self.diff_checkbox.isChecked())

    def display(self, diff_only):
        """
        Parameters
        ----------
        diff_only: bool
            Whether or not to show only differences.
        """
        array = self.array
        bg_value = self.bg_value
        if diff_only and self.isequal.ndim > 0:
            row_filter = (~self.isequal).any(self.stack_axis)
            array = array[row_filter]
            bg_value = bg_value[row_filter]
        self.arraywidget.set_data(array, bg_value=bg_value)


class ArrayComparator(AbstractEditor):
    """Array Comparator Dialog"""

    name = "Array Comparator"

    def __init__(self, parent=None):
        AbstractEditor.__init__(self, parent, editable=False, file_menu=False, help_menu=True)
        self.setup_menu_bar()

    def _setup_and_check(self, widget, data, title, readonly, **kwargs):
        """
        Setup ArrayComparator.

        Parameters
        ----------
        widget: QWidget
            Parent widget.
        data: dict of Array
            Arrays to compare as a {name: Array} dict.
        title: str
            Title.
        readonly: bool
            Ignored argument (comparator is always read only)
        kwargs:

          * rtol: int or float
          * atol: int or float
          * nans_equal: bool
          * bg_gradient: str
        """
        if isinstance(data, (list, tuple)):
            names = kwargs.pop('names', [f"Array{i}" for i in range(len(data))])
            data = dict(zip(names, data))
            warnings.warn("For ArrayComparator.setup_and_check, using a list or tuple for the data argument, "
                          "and using the names argument are both deprecated. Please use a dict instead",
                          FutureWarning, stacklevel=3)

        assert all(isinstance(s, la.Array) for s in data.values())

        layout = QVBoxLayout()
        widget.setLayout(layout)

        comparator_widget = ComparatorWidget(self, **kwargs)
        comparator_widget.set_data(data.values(), la.Axis(data.keys(), 'array'))
        layout.addWidget(comparator_widget)


class SessionComparator(AbstractEditor):
    """Session Comparator Dialog"""

    name = "Session Comparator"

    def __init__(self, parent=None):
        AbstractEditor.__init__(self, parent, editable=False, file_menu=False, help_menu=True)
        self.setup_menu_bar()

        self.sessions = None
        self.stack_axis = None
        self.listwidget = None

    def _setup_and_check(self, widget, data, title, readonly, **kwargs):
        """
        Setup SessionComparator.

        Parameters
        ----------
        widget: QWidget
            Parent widget.
        data: dict of Session
            Sessions to compare as a {name: Session} dict.
        title: str
            Title.
        readonly: bool
            Ignored argument (comparator is always read only)
        kwargs:

          * rtol: int or float
          * atol: int or float
          * nans_equal: bool
          * bg_gradient: str
        """
        if isinstance(data, (list, tuple)):
            names = kwargs.pop('names', [f"Session{i}" for i in range(len(data))])
            data = dict(zip(names, data))
            warnings.warn("For SessionComparator.setup_and_check, using a list or tuple for the data argument, "
                          "and using the names argument are both deprecated. Please use a dict instead",
                          FutureWarning, stacklevel=3)

        assert all(isinstance(s, la.Session) for s in data.values())
        self.sessions = data.values()
        self.stack_axis = la.Axis(data.keys(), 'session')

        layout = QVBoxLayout()
        widget.setLayout(layout)

        array_names = sorted(set.union(*[set(s.filter(kind=DISPLAY_IN_GRID).names) for s in self.sessions]))
        listwidget = QListWidget(self)
        listwidget.addItems(array_names)
        listwidget.currentItemChanged.connect(self.on_item_changed)
        for i, name in enumerate(array_names):
            arrays = self.get_arrays(name)
            first_array = arrays[0]
            if not all(a.equals(first_array, nans_equal=True) for a in arrays[1:]):
                listwidget.item(i).setForeground(Qt.red)
        self.listwidget = listwidget

        comparatorwidget = ComparatorWidget(self, **kwargs)
        # do not call set_data on the comparatorwidget as it will be done by the setCurrentRow below
        self.arraywidget = comparatorwidget

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(listwidget)
        main_splitter.addWidget(comparatorwidget)
        main_splitter.setSizes([5, 95])
        main_splitter.setCollapsible(1, False)
        self.widget_state_settings['main_splitter'] = main_splitter

        layout.addWidget(main_splitter)
        self.listwidget.setCurrentRow(0)

    def get_arrays(self, name):
        return [la.asarray(s.get(name, la.nan)) for s in self.sessions]

    def on_item_changed(self, curr, prev):
        arrays = self.get_arrays(str(curr.text()))
        self.arraywidget.set_data(arrays, self.stack_axis)

    def closeEvent(self, event):
        self.save_widgets_state_and_geometry()
        AbstractEditor.closeEvent(self, event)
