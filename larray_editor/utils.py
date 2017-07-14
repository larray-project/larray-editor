from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np

PY2 = sys.version[0] == '2'

from qtpy import PYQT5
from qtpy.QtCore import QVariant
from qtpy.QtGui import QIcon, QColor, QFont, QKeySequence
from qtpy.QtWidgets import QAction, QDialog, QVBoxLayout

if PYQT5:
    from matplotlib.backends.backend_qt5agg import FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
else:
    from matplotlib.backends.backend_qt4agg import FigureCanvas
    from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar


# Note: string and unicode data types will be formatted with '%s' (see below)
SUPPORTED_FORMATS = {
    'object': '%s',
    'single': '%.2f',
    'double': '%.2f',
    'float_': '%.2f',
    'longfloat': '%.2f',
    'float32': '%.2f',
    'float64': '%.2f',
    'float96': '%.2f',
    'float128': '%.2f',
    'csingle': '%r',
    'complex_': '%r',
    'clongfloat': '%r',
    'complex64': '%r',
    'complex128': '%r',
    'complex192': '%r',
    'complex256': '%r',
    'byte': '%d',
    'short': '%d',
    'intc': '%d',
    'int_': '%d',
    'longlong': '%d',
    'intp': '%d',
    'int8': '%d',
    'int16': '%d',
    'int32': '%d',
    'int64': '%d',
    'ubyte': '%d',
    'ushort': '%d',
    'uintc': '%d',
    'uint': '%d',
    'ulonglong': '%d',
    'uintp': '%d',
    'uint8': '%d',
    'uint16': '%d',
    'uint32': '%d',
    'uint64': '%d',
    'bool_': '%r',
    'bool8': '%r',
    'bool': '%r',
}

def _get_font(family, size, bold=False, italic=False):
    weight = QFont.Bold if bold else QFont.Normal
    font = QFont(family, size, weight)
    if italic:
        font.setItalic(True)
    return to_qvariant(font)

def is_float(dtype):
    """Return True if datatype dtype is a float kind"""
    return ('float' in dtype.name) or dtype.name in ['single', 'double']

def is_number(dtype):
    """Return True is datatype dtype is a number kind"""
    return is_float(dtype) or ('int' in dtype.name) or ('long' in dtype.name) or ('short' in dtype.name)

def get_font(section):
    return _get_font('Calibri', 11)

def to_qvariant(obj=None):
    return obj

def from_qvariant(qobj=None, pytype=None):
    # FIXME: force API level 2 instead of handling this
    if isinstance(qobj, QVariant):
        assert pytype is str
        return pytype(qobj.toString())
    return qobj

def _(text):
    return text

def to_text_string(obj, encoding=None):
    """Convert `obj` to (unicode) text string"""
    if PY2:
        # Python 2
        if encoding is None:
            return unicode(obj)
        else:
            return unicode(obj, encoding)
    else:
        # Python 3
        if encoding is None:
            return str(obj)
        elif isinstance(obj, str):
            # In case this function is not used properly, this could happen
            return obj
        else:
            return str(obj, encoding)

def keybinding(attr):
    """Return keybinding"""
    ks = getattr(QKeySequence, attr)
    return QKeySequence.keyBindings(ks)[0]

def create_action(parent, text, icon=None, triggered=None, shortcut=None, statustip=None):
    """Create a QAction"""
    action = QAction(text, parent)
    if triggered is not None:
        action.triggered.connect(triggered)
    if icon is not None:
        action.setIcon(icon)
    if shortcut is not None:
        action.setShortcut(shortcut)
    if statustip is not None:
        action.setStatusTip(statustip)
    # action.setShortcutContext(Qt.WidgetShortcut)
    return action

def clear_layout(layout):
    for i in reversed(range(layout.count())):
        item = layout.itemAt(i)
        widget = item.widget()
        if widget is not None:
            # widget.setParent(None)
            widget.deleteLater()
        layout.removeItem(item)

def get_idx_rect(index_list):
    """Extract the boundaries from a list of indexes"""
    rows = [i.row() for i in index_list]
    cols = [i.column() for i in index_list]
    return min(rows), max(rows), min(cols), max(cols)


class IconManager(object):

    _icons = {'larray': 'larray.ico'}
    _icon_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')

    def icon(self, ref):
        if ref in self._icons:
            icon_path = os.path.join(self._icon_dir, self._icons[ref])
            return QIcon(icon_path)
        else:
        # By default, only X11 will support themed icons. In order to use
        # themed icons on Mac and Windows, you will have to bundle a compliant
        # theme in one of your PySide.QtGui.QIcon.themeSearchPaths() and set the
        # appropriate PySide.QtGui.QIcon.themeName() .
            return QIcon.fromTheme(ref)

ima = IconManager()


class LinearGradient(object):
    """
    I cannot believe I had to roll my own class for this when PyQt already
    contains QLinearGradient... but you cannot get intermediate values out of
    QLinearGradient!
    """
    def __init__(self, stop_points=None):
        if stop_points is None:
            stop_points = []
        # sort by position
        stop_points = sorted(stop_points, key=lambda x: x[0])
        positions, colors = zip(*stop_points)
        self.positions = np.array(positions)
        assert len(np.unique(self.positions)) == len(self.positions)
        self.colors = np.array(colors)

    def __getitem__(self, key):
        """
        Parameters
        ----------
        key : float

        Returns
        -------
        QColor
        """
        if key != key:
            key = self.positions[0]
        pos_idx = np.searchsorted(self.positions, key, side='right') - 1
        # if we are exactly on one of the bounds
        if pos_idx > 0 and key in self.positions:
            pos_idx -= 1
        pos0, pos1 = self.positions[pos_idx:pos_idx + 2]
        # col0 and col1 are ndarrays
        col0, col1 = self.colors[pos_idx:pos_idx + 2]
        assert pos0 != pos1
        color = col0 + (col1 - col0) * (key - pos0) / (pos1 - pos0)
        return to_qvariant(QColor.fromHsvF(*color))


class PlotDialog(QDialog):
    def __init__(self, canvas, parent=None):
        super(PlotDialog, self).__init__(parent)

        toolbar = NavigationToolbar(canvas, self)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        self.setLayout(layout)
        canvas.draw()

def show_figure(parent, figure):
    canvas = FigureCanvas(figure)
    main = PlotDialog(canvas, parent)
    main.show()


class Product(object):
    """
    Represents the `cartesian product` of several arrays.

    Parameters
    ----------
    arrays : iterable of array
        List of arrays on which to apply the cartesian product.

    Examples
    --------
    >>> p = Product([['a', 'b', 'c'], [1, 2]])
    >>> for i in range(len(p)):
    ...     print(p[i])
    ('a', 1)
    ('a', 2)
    ('b', 1)
    ('b', 2)
    ('c', 1)
    ('c', 2)
    >>> p[1:4]
    [('a', 2), ('b', 1), ('b', 2)]
    >>> list(p)
    [('a', 1), ('a', 2), ('b', 1), ('b', 2), ('c', 1), ('c', 2)]
    """
    def __init__(self, arrays):
        self.arrays = arrays
        assert len(arrays)
        shape = [len(a) for a in self.arrays]
        self.div_mod = [(int(np.prod(shape[i + 1:])), shape[i])
                        for i in range(len(shape))]
        self.length = np.prod(shape)

    def to_tuple(self, key):
        if key >= self.length:
            raise IndexError("index %d out of range for Product of length %d" % (key, self.length))
        return tuple(key // div % mod for div, mod in self.div_mod)

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return tuple(array[i]
                         for array, i in zip(self.arrays, self.to_tuple(key)))
        else:
            assert isinstance(key, slice), \
                "key (%s) has invalid type (%s)" % (key, type(key))
            start, stop, step = key.start, key.stop, key.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.length
            if step is None:
                step = 1

            return [tuple(array[i]
                          for array, i in zip(self.arrays, self.to_tuple(i)))
                    for i in range(start, stop, step)]


class _LazyLabels(object):
    def __init__(self, arrays):
        self.prod = Product(arrays)

    def __getitem__(self, key):
        return ' '.join(self.prod[key])

    def __len__(self):
        return len(self.prod)


class _LazyDimLabels(object):
    """
    Examples
    --------
    >>> p = Product([['a', 'b', 'c'], [1, 2]])
    >>> list(p)
    [('a', 1), ('a', 2), ('b', 1), ('b', 2), ('c', 1), ('c', 2)]
    >>> l0 = _LazyDimLabels(p, 0)
    >>> l1 = _LazyDimLabels(p, 1)
    >>> for i in range(len(p)):
    ...     print(l0[i], l1[i])
    a 1
    a 2
    b 1
    b 2
    c 1
    c 2
    >>> l0[1:4]
    ['a', 'b', 'b']
    >>> l1[1:4]
    [2, 1, 2]
    >>> list(l0)
    ['a', 'a', 'b', 'b', 'c', 'c']
    >>> list(l1)
    [1, 2, 1, 2, 1, 2]
    """
    def __init__(self, prod, i):
        self.prod = prod
        self.i = i

    def __iter__(self):
        return iter(self.prod[i][self.i] for i in range(len(self.prod)))

    def __getitem__(self, key):
        key_prod = self.prod[key]
        if isinstance(key, slice):
            return [p[self.i] for p in key_prod]
        else:
            return key_prod[self.i]

    def __len__(self):
        return len(self.prod)


class _LazyRange(object):
    def __init__(self, length, offset):
        self.length = length
        self.offset = offset

    def __getitem__(self, key):
        if key >= self.offset:
            return key - self.offset
        else:
            return ''

    def __len__(self):
        return self.length + self.offset


class _LazyNone(object):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, key):
        return ' '

    def __len__(self):
        return self.length
