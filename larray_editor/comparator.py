import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QSplitter, QDialogButtonBox, QHBoxLayout,
                            QDialog, QLabel, QCheckBox)

from larray import LArray, Session, Axis, X, stack, full_like, nan, zeros_like, isnan, larray_nan_equal, nan_equal
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

        self.arraywidget = ArrayEditorWidget(self, np.array([]), readonly=True, bg_gradient='red-white-blue')

        diff_checkbox = QCheckBox(_('Differences Only'))
        diff_checkbox.stateChanged.connect(self.show_differences_only)
        self.diff_checkbox = diff_checkbox
        self.arraywidget.btn_layout.addWidget(diff_checkbox)

        layout.addWidget(self.arraywidget)

        self.array = None
        self.isequal = None
        self.bg_value = None
        self.stack_axis = None

    def set_data(self, arrays, stack_axis):
        assert all(np.isscalar(a) or isinstance(a, LArray) for a in arrays)
        self.stack_axis = stack_axis
        try:
            self.array = stack(arrays, stack_axis)
            array0 = self.array[stack_axis.i[0]]
        except:
            self.array = LArray([np.nan])
            array0 = self.array
        try:
            self.isequal = nan_equal(self.array, array0)
        except TypeError:
            self.isequal = self.array == array0

        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                diff = self.array - array0
                reldiff = diff / array0
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
            maxabsreldiff = np.nan
            self.bg_value = full_like(self.array, 0.5)

        self.maxdiff_label.setText(str(maxabsreldiff))
        self.arraywidget.set_data(self.array, bg_value=self.bg_value)

    def show_differences_only(self, yes):
        if yes:
            # only show rows with a difference. For some reason, this is abysmally slow though.
            row_filter = (~self.isequal).any(self.stack_axis.name)
            self.arraywidget.set_data(self.array[row_filter], bg_value=self.bg_value[row_filter])
        else:
            self.arraywidget.set_data(self.array, bg_value=self.bg_value)


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
        comparatorwidget.set_data(self.get_arrays(array_names[0]), self.stack_axis)
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
        return True

    def get_arrays(self, name):
        return [s.get(name, nan) for s in self.sessions]

    def on_item_changed(self, curr, prev):
        arrays = self.get_arrays(str(curr.text()))
        self.comparatorwidget.set_data(arrays, self.stack_axis)
