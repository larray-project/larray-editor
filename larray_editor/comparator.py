import ast
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QSplitter, QDialogButtonBox, QHBoxLayout,
                            QDialog, QLabel, QCheckBox, QLineEdit, QComboBox)

from larray import LArray, Session, Axis, stack, full_like, nan, larray_nan_equal, element_equal
from larray_editor.utils import ima, replace_inf, _
from larray_editor.arraywidget import ArrayEditorWidget


class ComparatorWidget(QWidget):
    """Comparator Widget"""
    def __init__(self, parent=None):
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

        self.arraywidget = ArrayEditorWidget(self, data=None, readonly=True, bg_gradient='red-white-blue')

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

        # add local arraywidget to layout
        self.arraywidget.btn_layout.addStretch()
        layout.addWidget(self.arraywidget)

        self.array = None
        self.array0 = None
        self.isequal = None
        self.bg_value = None
        self.stack_axis = None

    # override keyPressEvent to prevent pressing Enter after changing the tolerance value
    # in associated QLineEdit to close the parent dialog box
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            return
        QWidget.keyPressEvent(self, event)

    def set_data(self, arrays, stack_axis):
        assert all(np.isscalar(a) or isinstance(a, LArray) for a in arrays)
        self.stack_axis = stack_axis
        try:
            self.array = stack(arrays, stack_axis)
            self.array0 = self.array[stack_axis.i[0]]
        except Exception as e:
            self.array = LArray(str(e))
            self.array0 = self.array
        self.update_isequal()

    def update_isequal(self):
        try:
            tol_str = self.tolerance_line_edit.text()
            tol = ast.literal_eval(tol_str) if tol_str else 0
            atol, rtol = (tol, 0) if self.tolerance_combobox.currentText() == "absolute" else (0, tol)
            self.isequal = element_equal(self.array, self.array0, rtol=rtol, atol=atol, nan_equals=True)
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
                self.bg_value = full_like(self.array, 0.5)
        except TypeError:
            # str/object array
            maxabsreldiff = nan
            self.bg_value = full_like(self.array, 0.5)

        self.maxdiff_label.setText(str(maxabsreldiff))
        self.display(self.diff_checkbox.isChecked())

    def display(self, diff_only):
        array = self.array
        bg_value = self.bg_value
        if diff_only and self.isequal.ndim > 0:
            row_filter = (~self.isequal).any(self.stack_axis)
            array = array[row_filter]
            bg_value = bg_value[row_filter]
        self.arraywidget.set_data(array, bg_value=bg_value)


class ArrayComparator(QDialog):
    """Array Comparator Dialog"""
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)

    def setup_and_check(self, arrays, names, title=''):
        """
        Setup ArrayComparator:
        return False if data is not supported, True otherwise
        """
        icon = ima.icon('larray')
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Array comparator")
        title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)

        layout = QVBoxLayout()
        comparator_widget = ComparatorWidget(self)
        comparator_widget.set_data(arrays, Axis(names, 'array'))
        layout.addWidget(comparator_widget)

        # Buttons configuration
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        buttons = QDialogButtonBox.Ok
        bbox = QDialogButtonBox(buttons)
        bbox.accepted.connect(self.accept)
        btn_layout.addWidget(bbox)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.resize(800, 600)
        self.setMinimumSize(400, 300)

        # Make the dialog act as a window
        self.setWindowFlags(Qt.Window)
        return True


# TODO: it should be possible to reuse both MappingEditor and ArrayComparator
class SessionComparator(QDialog):
    """Session Comparator Dialog"""
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.sessions = None
        self.stack_axis = None
        self.comparatorwidget = None
        self.listwidget = None

    def setup_and_check(self, sessions, names, title='', colors='red-white-blue'):
        """
        Setup SessionComparator:
        return False if data is not supported, True otherwise
        """
        assert all(isinstance(s, Session) for s in sessions)
        self.sessions = sessions
        self.stack_axis = Axis(names, 'session')

        icon = ima.icon('larray')
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Session comparator")
        title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)

        layout = QVBoxLayout()
        self.setLayout(layout)

        array_names = sorted(set.union(*[set(s.names) for s in self.sessions]))
        listwidget = QListWidget(self)
        listwidget.addItems(array_names)
        listwidget.currentItemChanged.connect(self.on_item_changed)
        for i, name in enumerate(array_names):
            arrays = self.get_arrays(name)
            if not all(larray_nan_equal(a, arrays[0]) for a in arrays[1:]):
                listwidget.item(i).setForeground(Qt.red)
        self.listwidget = listwidget

        comparatorwidget = ComparatorWidget(self)
        self.comparatorwidget = comparatorwidget

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(listwidget)
        main_splitter.addWidget(comparatorwidget)
        main_splitter.setSizes([5, 95])
        main_splitter.setCollapsible(1, False)

        layout.addWidget(main_splitter)

        # Buttons configuration
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        buttons = QDialogButtonBox.Ok
        bbox = QDialogButtonBox(buttons)
        bbox.accepted.connect(self.accept)
        btn_layout.addWidget(bbox)
        layout.addLayout(btn_layout)

        self.resize(800, 600)
        self.setMinimumSize(400, 300)

        # Make the dialog act as a window
        self.setWindowFlags(Qt.Window)
        self.listwidget.setCurrentRow(0)
        return True

    def get_arrays(self, name):
        return [s.get(name, nan) for s in self.sessions]

    def on_item_changed(self, curr, prev):
        arrays = self.get_arrays(str(curr.text()))
        self.comparatorwidget.set_data(arrays, self.stack_axis)
