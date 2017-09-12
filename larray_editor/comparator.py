from larray import LArray, Session, Axis, x, stack, full_like, larray_equal
from larray_editor.utils import ima, LinearGradient, _
from larray_editor.arraywidget import ArrayEditorWidget

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QSplitter, QDialogButtonBox, QHBoxLayout,
                            QDialog, QLabel)


class ArrayComparator(QDialog):
    """Session Editor Dialog"""
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.arrays = None
        self.array = None
        self.arraywidget = None

    def setup_and_check(self, arrays, names, title=''):
        """
        Setup ArrayComparator:
        return False if data is not supported, True otherwise
        """
        assert all(isinstance(a, LArray) for a in arrays)
        self.arrays = arrays
        self.array = stack(arrays, Axis(names, 'arrays'))

        icon = ima.icon('larray')
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Array comparator")
        title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)

        layout = QVBoxLayout()
        self.setLayout(layout)

        diff = self.array - self.array[x.arrays.i[0]]
        absmax = abs(diff).max()

        # max diff label
        maxdiff_layout = QHBoxLayout()
        maxdiff_layout.addWidget(QLabel('maximum absolute difference: ' +
                                        str(absmax)))
        maxdiff_layout.addStretch()
        layout.addLayout(maxdiff_layout)

        if absmax:
            # scale diff to range 0-1
            bg_value = (diff / absmax) / 2 + 0.5
        else:
            # all 0.5 (white)
            bg_value = full_like(diff, 0.5)
        gradient = LinearGradient([(0, [.99, .85, 1., .6]),
                                   (0.5 - 1e-16, [.99, .15, 1., .6]),
                                   (0.5, [1., 0., 1., 1.]),
                                   (0.5 + 1e-16, [.66, .15, 1., .6]),
                                   (1, [.66, .85, 1., .6])])

        self.arraywidget = ArrayEditorWidget(self, self.array, readonly=True,
                                             bg_value=bg_value,
                                             bg_gradient=gradient)

        layout.addWidget(self.arraywidget)

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
        self.names = None
        self.arraywidget = None
        self.maxdiff_label = None
        self.gradient = LinearGradient([(0, [.99, .85, 1., .6]),
                                        (0.5 - 1e-16, [.99, .15, 1., .6]),
                                        (0.5, [1., 0., 1., 1.]),
                                        (0.5 + 1e-16, [.66, .15, 1., .6]),
                                        (1, [.66, .85, 1., .6])])

    def setup_and_check(self, sessions, names, title=''):
        """
        Setup SessionComparator:
        return False if data is not supported, True otherwise
        """
        assert all(isinstance(s, Session) for s in sessions)
        self.sessions = sessions
        self.names = names

        icon = ima.icon('larray')
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Session comparator")
        title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)

        layout = QVBoxLayout()
        self.setLayout(layout)

        names = sorted(set.union(*[set(s.names) for s in self.sessions]))
        self._listwidget = listwidget = QListWidget(self)
        self._listwidget.addItems(names)
        self._listwidget.currentItemChanged.connect(self.on_item_changed)

        for i, name in enumerate(names):
            arrays = [s.get(name) for s in self.sessions]
            eq = [larray_equal(a, arrays[0]) for a in arrays[1:]]
            if not all(eq):
                listwidget.item(i).setForeground(Qt.red)

        array, absmax, bg_value = self.get_array(names[0])

        if not array.size:
            array = LArray(['no data'])
        self.arraywidget = ArrayEditorWidget(self, array, readonly=True,
                                             bg_value=bg_value,
                                             bg_gradient=self.gradient)

        right_panel_layout = QVBoxLayout()

        # max diff label
        maxdiff_layout = QHBoxLayout()
        maxdiff_layout.addWidget(QLabel('maximum absolute difference:'))
        self.maxdiff_label = QLabel(str(absmax))
        maxdiff_layout.addWidget(self.maxdiff_label)
        maxdiff_layout.addStretch()
        right_panel_layout.addLayout(maxdiff_layout)

        # array_splitter.setSizePolicy(QSizePolicy.Expanding,
        #                              QSizePolicy.Expanding)
        right_panel_layout.addWidget(self.arraywidget)

        # you cant add a layout directly in a splitter, so we have to wrap it
        # in a widget
        right_panel_widget = QWidget()
        right_panel_widget.setLayout(right_panel_layout)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self._listwidget)
        main_splitter.addWidget(right_panel_widget)
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

    def get_array(self, name):
        arrays = [s.get(name) for s in self.sessions]
        array = stack(arrays, Axis(self.names, 'sessions'))
        diff = array - array[x.sessions.i[0]]
        absmax = abs(diff).max()
        # scale diff to 0-1
        if absmax:
            bg_value = (diff / absmax) / 2 + 0.5
        else:
            bg_value = full_like(diff, 0.5)
        # only show rows with a difference. For some reason, this is abysmally
        # slow though.
        # row_filter = (array != array[la.x.sessions.i[0]]).any(la.x.sessions)
        # array = array[row_filter]
        # bg_value = bg_value[row_filter]
        return array, absmax, bg_value

    def on_item_changed(self, curr, prev):
        array, absmax, bg_value = self.get_array(str(curr.text()))
        self.maxdiff_label.setText(str(absmax))
        self.arraywidget.set_data(array, bg_value=bg_value,
                                  bg_gradient=self.gradient)
