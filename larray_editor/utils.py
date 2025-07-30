import os
import signal
import socket
import sys
import math
import logging
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Union

from larray.util.misc import Product

import numpy as np
try:
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass

from qtpy import QtCore
from qtpy.QtCore import Qt, QSettings
from qtpy.QtGui import QIcon, QColor, QFont, QKeySequence, QLinearGradient
from qtpy.QtWidgets import QAction, QDialog, QVBoxLayout

try:
    # try the un-versioned backend first (for matplotlib 3.5+)

    # this is equivalent to "from matplotlib.backends.backend_qtagg import FigureCanvas" but is easier to statically
    # analyze for PyCharm et al.
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    # fall back to explicit qt5 backend (for matplotlib < 3.5)
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

logger = logging.getLogger("editor")


CORE_DEPENDENCIES = ['matplotlib', 'numpy', 'openpyxl', 'pandas', 'pytables', 'xlsxwriter', 'xlrd', 'xlwings']
EDITOR_DEPENDENCIES = ['larray', 'larray_eurostat', 'qt'] + CORE_DEPENDENCIES
EUROSTAT_DEPENDENCIES = ['larray']
DEPENDENCIES = {'editor': EDITOR_DEPENDENCIES, 'larray': CORE_DEPENDENCIES, 'larray_eurostat': EUROSTAT_DEPENDENCIES}


DOC = "http://larray.readthedocs.io/en/{version}"
URLS = {"fpb": "http://www.plan.be/index.php?lang=en",
        "GPL3": "https://www.gnu.org/licenses/gpl-3.0.html",
        "doc_index": f"{DOC}/index.html",
        "doc_tutorial": f"{DOC}/tutorial.html",
        "doc_api": f"{DOC}/api.html",
        "new_issue_editor": "https://github.com/larray-project/larray-editor/issues/new",
        "new_issue_larray": "https://github.com/larray-project/larray/issues/new",
        "new_issue_larray_eurostat": "https://github.com/larray-project/larray_eurostat/issues/new",
        "announce_group": "https://groups.google.com/d/forum/larray-announce",
        "users_group": "https://groups.google.com/d/forum/larray-users"}

commonpath = os.path.commonpath


def get_module_version(module_name):
    """Return the version of a module if installed, N/A otherwise"""
    try:
        from importlib import import_module
        module = import_module(module_name)
        if 'qtpy' in module_name:
            from qtpy import API_NAME, PYQT_VERSION  # API_NAME --> PyQt5 or PyQt4
            qt_version = module.__version__
            return f'{qt_version}, {API_NAME} {PYQT_VERSION}'
        # at least for matplotlib, we cannot test this using '__version__' in dir(module)
        elif hasattr(module, '__version__'):
            return module.__version__
        elif hasattr(module, '__VERSION__'):
            return module.__VERSION__
        else:
            return 'N/A'
    except ImportError:
        return 'N/A'


def get_versions(package):
    """Get version information of dependencies of one of our packages
    `package` can be one of 'editor', 'larray' or 'larray_eurostat'
    """
    import platform
    module_with_version = {'editor': 'larray_editor', 'qt': 'qtpy.QtCore', 'pytables': 'tables'}

    versions = {
        'system': platform.system() if sys.platform != 'darwin' else 'Darwin',
        'python': platform.python_version(),
        'bitness': 64 if sys.maxsize > 2**32 else 32,
    }

    versions[package] = get_module_version(module_with_version.get(package, package))
    for dep in DEPENDENCIES[package]:
        versions[dep] = get_module_version(module_with_version.get(dep, dep))

    return versions


def get_documentation_url(key):
    version = get_module_version('larray')
    if version == 'N/A':
        version = 'stable'
    return URLS[key].format(version=version)


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
    return font


def is_float(dtype):
    """Return True if datatype dtype is a float kind"""
    return ('float' in dtype.name) or dtype.name in ['single', 'double']


def is_number(dtype):
    """Return True is datatype dtype is a number kind"""
    return is_float(dtype) or ('int' in dtype.name) or ('long' in dtype.name) or ('short' in dtype.name)


# When we drop support for Python3.9, we can use traceback.print_exception
def print_exception(exception):
    type_ = type(exception)
    traceback.print_exception(type_, exception, exception.__traceback__)


def format_exception(exception):
    type_ = type(exception)
    return traceback.format_exception(type_, exception, exception.__traceback__)


def get_default_font():
    return _get_font('Calibri', 11)


def _(text):
    return text


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


class IconManager:
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
from_hsvf = np.frompyfunc(QColor.fromHsvF, 4, 1)


class LinearGradient:
    """
    I cannot believe I had to roll my own class for this when PyQt already
    contains QLinearGradient... but you cannot get intermediate values out of
    QLinearGradient!

    Parameters
    ----------
    stop_points: list/tuple, optional
        List containing pairs (stop_position, colors_HsvF).
        `colors` is a 4 elements list containing `hue`, `saturation`, `value` and `alpha-channel`
    """
    def __init__(self, stop_points=None, nan_color=None):
        if stop_points is None:
            stop_points = []
        # sort by position
        stop_points = sorted(stop_points, key=lambda x: x[0])
        positions, colors = zip(*stop_points)
        positions = np.array(positions)
        # check positions are unique and between 0 and 1
        assert len(np.unique(positions)) == len(positions)
        assert np.all((0 <= positions) & (positions <= 1))
        assert positions[0] == 0
        assert positions[-1] == 1
        self.positions = positions
        self.colors = np.array(colors)
        if nan_color is None:
            nan_color = QColor(Qt.gray)
        self.nan_color = nan_color

    def as_qgradient(self):
        qgradient = QLinearGradient(0, 0, 100, 0)
        for pos, color in zip(self.positions, self.colors):
            qgradient.setColorAt(pos, QColor.fromHsvF(*color))
        return qgradient

    def __getitem__(self, key):
        """
        Parameters
        ----------
        key : float or array-like of floats
            must be between 0 and 1 to return a color from the gradient.
            Otherwise, will return nan_color.

        Returns
        -------
        QColor
        """
        if np.isscalar(key):
            # the scalar code path is only used if an adapter returns a single color
            if np.isnan(key) or key < 0 or key > 1:
                return self.nan_color
            pos_idx = np.searchsorted(self.positions, key, side='right') - 1
            # if we are exactly on the last bound
            if key == 1:  # self.positions[-1] is always 1
                # for other bounds, this is not a problem because if we use the
                # "end" of the "previous" color or the "start" of the "next",
                # this should be the same
                assert pos_idx == len(self.positions) - 1
                pos_idx -= 1
            pos0, pos1 = self.positions[pos_idx:pos_idx + 2]
            assert pos1 > pos0
            # color0 and color1 are ndarrays
            color0, color1 = self.colors[pos_idx:pos_idx + 2]
            color = color0 + (color1 - color0) * (key - pos0) / (pos1 - pos0)
            return QColor.fromHsvF(*color)
        else:
            key = np.asarray(key)

            # this should never happen, but lets be robust to buggy code
            key = np.where((key < 0) | (key > 1), np.nan, key)

            # for positions [0, 0.5, 1], this will yield:
            # key          | pos_idx | pos_idx + 1
            # -------------|---------|------------
            # < 0          |      -1 |           0
            # >= 0 & < 0.5 |       0 |           1
            # >= 0.5 & < 1 |       1 |           2
            # >= 1         |       2 |           3
            # nan          |       2 |           3
            pos_idx = np.searchsorted(self.positions, key, side='right') - 1
            pos0 = self.positions[pos_idx]
            # if we are exactly on the last bound (key == 1) or key is nan,
            # pos_idx + 1 will be out of bounds, but mode='clip' handles that for us
            pos1 = self.positions.take(pos_idx + 1, mode='clip')
            # color0 and color1 are ndarrays
            color0 = self.colors[pos_idx]
            color1 = self.colors.take(pos_idx + 1, axis=0, mode='clip')
            div = np.where(pos1 != pos0, pos1 - pos0, 1)
            normalized_value = (key - pos0) / div
            # newaxis so that we can broadcast with the 4 color channels
            key_isnan = np.isnan(key)[..., np.newaxis]
            color = color0 + (color1 - color0) * normalized_value[..., np.newaxis]
            color = np.where(key_isnan, self.nan_color.getHsvF(), color)
            return from_hsvf(color[..., 0], color[..., 1], color[..., 2], color[..., 3])


class PlotDialog(QDialog):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)

        toolbar = NavigationToolbar(canvas, self)

        # set the layout
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        self.setLayout(layout)
        canvas.draw()


def show_figure(parent, figure, title=None):
    if (figure.canvas is not None and figure.canvas.manager is not None and
            figure.canvas.manager.window is not None):
        figure.canvas.draw()
        window = figure.canvas.manager.window
        window.raise_()
    else:
        canvas = FigureCanvas(figure)
        window = PlotDialog(canvas, parent)
    if title is not None:
        window.setWindowTitle(title)
    window.show()


class Axis:
    """
    Represents an Axis.

    Parameters
    ----------
    id : str or int
        Id of axis.
    name : str
        Name of the axis. Can be None.
    labels : list or tuple or 1D array
        List of labels
    """
    def __init__(self, id, name, labels):
        self.id = id
        self.name = name
        self.labels = labels

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        if not isinstance(id, (str, int)):
            raise TypeError("id must a string or a integer")
        self._id = id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self._name = name

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        if not (hasattr(labels, '__len__') and hasattr(labels, '__getitem__')):
            raise TypeError("labels must be a list or tuple or any 1D array-like")
        self._labels = labels

    def __len__(self):
        return len(self.labels)

    def __str__(self):
        return f'Axis({self.id}, {self.name}, {self.labels})'


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


def replace_inf(value):
    """Replace -inf/+inf in array with respectively min(array_without_inf)/max(array_without_inf).

    It leaves nans intact.

    Parameters
    ----------
    value : np.ndarray or any compatible type
        Input array.

    Returns
    -------
    (np.ndarray, float, float)
        array with infinite values replaced by the min and maximum respectively
        minimum finite value
        maximum finite value

    Examples
    --------
    >>> replace_inf(np.array([-5, np.inf, 0, -np.inf, -4, np.nan, 5]))
    (array([ -5.,   5.,   0.,  -5.,  -4.,  nan,   5.]), -5.0, 5.0)
    """
    value = value.copy()
    # replace -inf by min(value)
    isneginf = value == -np.inf
    minvalue = np.nanmin(value[~isneginf])
    value[isneginf] = minvalue
    # replace +inf by max(value)
    isposinf = value == np.inf
    maxvalue = np.nanmax(value[~isposinf])
    value[isposinf] = maxvalue
    return value, minvalue, maxvalue


def scale_to_01range(value, vmin, vmax):
    """Scale value to 0-1 range based on vmin and vmax.

    NaN are left intact, but -inf and +inf are converted to 0 and 1 respectively.

    Parameters
    ----------
    value : any numeric type
        Value to scale.
    vmin : any numeric type
        Minimum used to do the scaling. This is the minimum value that is valid for value, *excluding -inf*.
        vmin must be <= vmax. vmin and vmax can be `nan`, in which case scale_to_01range will return `nan`.
    vmax : any numeric type
        Maximum used to do the scaling. This is the maximum value that is valid for value, *excluding +inf*.
        vmax must be >= vmin. vmin and vmax can be `nan`, in which case scale_to_01range will return `nan`.

    Returns
    -------
    float or np.ndarray

    Examples
    --------
    >>> scale_to_01range(5, 0, 10)
    0.5
    >>> scale_to_01range(1, 0, 10)
    0.1
    >>> scale_to_01range(np.nan, 0, 10)
    nan
    >>> scale_to_01range(+np.inf, 0, 10)
    1.0
    >>> scale_to_01range(-np.inf, 0, 10)
    0.0
    >>> scale_to_01range(5, 5, 5)
    0.0
    >>> scale_to_01range(np.array([-5, np.inf, 0, -np.inf, -4, 5]), -5, 5)
    array([ 0. ,  1. ,  0.5,  0. ,  0.1,  1. ])
    """
    if hasattr(value, 'shape') and value.shape:
        if np.isnan(vmin) or np.isnan(vmax) or (vmin == vmax):
            return np.where(np.isnan(value), np.nan, 0)
        else:
            assert vmin < vmax, f"vmin ({vmin}) < vmax ({vmax})"
            with np.errstate(divide='ignore', invalid='ignore'):
                res = (value - vmin) / (vmax - vmin)
                res[value == -np.inf] = 0
                res[value == +np.inf] = 1
            return res
    else:
        if np.isnan(value):
            return np.nan
        elif value == -np.inf:
            return 0.0
        elif value == +np.inf:
            return 1.0
        elif np.isnan(vmin) or np.isnan(vmax) or (vmin == vmax):
            return 0.0
        else:
            assert vmin < vmax
            return (value - vmin) / (vmax - vmin)


is_number_value = np.vectorize(lambda x: isinstance(x, (int, float, np.number)))


def get_sample_step(data, maxsize):
    size = data.size
    if not size:
        return None
    return int(math.ceil(size / maxsize))


def get_sample(data, maxsize):
    """return sample. View in all cases.

    if data.size < maxsize:
        sample_size == data.size
    else:
        (maxsize // 2) < sample_size <= maxsize

    Parameters
    ----------
    data
    maxsize

    Returns
    -------
    view
    """
    size = data.size
    if not size:
        return data
    return data.flat[::get_sample_step(data, maxsize)]


def get_sample_indices(data, maxsize):
    flat_indices = np.arange(0, data.size, get_sample_step(data, maxsize))
    return np.unravel_index(flat_indices, data.shape)


class RecentlyUsedList:
    MAX_RECENT_FILES = 10

    def __init__(self, list_name, parent_action=None, triggered=None):
        self.settings = QSettings()
        self.list_name = list_name
        if self.settings.value(list_name) is None:
            self.settings.setValue(list_name, [])
        if parent_action is not None:
            actions = [QAction(parent_action) for _ in range(self.MAX_RECENT_FILES)]
            for action in actions:
                action.setVisible(False)
                if triggered is not None:
                    action.triggered.connect(triggered)
            self._actions = actions
        else:
            self._actions = None
        self._update_actions()

    @property
    def files(self):
        return self.settings.value(self.list_name)

    @files.setter
    def files(self, files):
        self.settings.setValue(self.list_name, files[:self.MAX_RECENT_FILES])
        self._update_actions()

    @property
    def actions(self):
        return self._actions

    def add(self, filepath: Union[str, Path]):
        if filepath is None:
            return
        elif isinstance(filepath, Path):
            filepath = str(filepath)
        recent_files = self.files
        if filepath in recent_files:
            recent_files.remove(filepath)
        recent_files = [filepath] + recent_files
        self.files = recent_files

    def clear(self):
        self.files = []

    def _update_actions(self):
        if self.actions is not None:
            recent_files = self.files
            if recent_files is None:
                recent_files = []

            # zip will iterate up to the shortest of the two
            for filepath, action in zip(recent_files, self.actions):
                if isinstance(filepath, Path):
                    filepath = str(filepath)
                action.setText(os.path.basename(filepath))
                action.setStatusTip(filepath)
                action.setData(filepath)
                action.setVisible(True)
            # if we have less recent files than actions, hide the remaining actions
            for action in self.actions[len(recent_files):]:
                action.setVisible(False)


def cached_property(must_invalidate_cache_method):
    """A decorator to cache class properties."""
    def getter_decorator(original_getter):
        def caching_getter(self):
            if must_invalidate_cache_method(self) or not hasattr(self, '_cached_property_values'):
                self._cached_property_values = {}
            try:
                # cache hit
                return self._cached_property_values[original_getter]
            except KeyError:
                # property not computed yet (cache miss)
                value = original_getter(self)
                self._cached_property_values[original_getter] = value
                return value
        return property(caching_getter)
    return getter_decorator


# The following two functions (_allow_interrupt and _allow_interrupt_qt) are
# copied from matplotlib code, because they are not part of their public API
# and relying on them would require us to pin the matplotlib version, which
# is impractical.

# They are thus both governed by the matplotlib license at:
# https://github.com/matplotlib/matplotlib/blob/
#     2b3eea58ab653bea0ee35c1e93a481acd9f1b8d0/LICENSE/LICENSE

# Function copied from:
# https://github.com/matplotlib/matplotlib/blob/
#     2b3eea58ab653bea0ee35c1e93a481acd9f1b8d0/lib/matplotlib/backend_bases.py
@contextmanager
def _allow_interrupt(prepare_notifier, handle_sigint):
    """
    A context manager that allows terminating a plot by sending a SIGINT.  It
    is necessary because the running backend prevents the Python interpreter
    from running and processing signals (i.e., to raise a KeyboardInterrupt).
    To solve this, one needs to somehow wake up the interpreter and make it
    close the plot window.  We do this by using the signal.set_wakeup_fd()
    function which organizes a write of the signal number into a socketpair.
    A backend-specific function, *prepare_notifier*, arranges to listen to
    the pair's read socket while the event loop is running.  (If it returns a
    notifier object, that object is kept alive while the context manager runs.)

    If SIGINT was indeed caught, after exiting the on_signal() function the
    interpreter reacts to the signal according to the handler function which
    had been set up by a signal.signal() call; here, we arrange to call the
    backend-specific *handle_sigint* function.  Finally, we call the old SIGINT
    handler with the same arguments that were given to our custom handler.

    We do this only if the old handler for SIGINT was not None, which means
    that a non-python handler was installed, i.e. in Julia, and not SIG_IGN
    which means we should ignore the interrupts.

    Parameters
    ----------
    prepare_notifier : Callable[[socket.socket], object]
    handle_sigint : Callable[[], object]
    """

    old_sigint_handler = signal.getsignal(signal.SIGINT)
    if old_sigint_handler in (None, signal.SIG_IGN, signal.SIG_DFL):
        yield
        return

    handler_args = None
    wsock, rsock = socket.socketpair()
    wsock.setblocking(False)
    rsock.setblocking(False)
    old_wakeup_fd = signal.set_wakeup_fd(wsock.fileno())
    _ = prepare_notifier(rsock)

    def save_args_and_handle_sigint(*args):
        nonlocal handler_args
        handler_args = args
        handle_sigint()

    signal.signal(signal.SIGINT, save_args_and_handle_sigint)
    try:
        yield
    finally:
        wsock.close()
        rsock.close()
        signal.set_wakeup_fd(old_wakeup_fd)
        signal.signal(signal.SIGINT, old_sigint_handler)
        if handler_args is not None:
            old_sigint_handler(*handler_args)


# Function copied from:
# https://github.com/matplotlib/matplotlib/blob/
#     2b3eea58ab653bea0ee35c1e93a481acd9f1b8d0/lib/matplotlib/backends/backend_qt.py
def _allow_interrupt_qt(qapp_or_eventloop):
    """A context manager that allows terminating a plot by sending a SIGINT."""

    # Use QSocketNotifier to read the socketpair while the Qt event loop runs.

    def prepare_notifier(rsock):
        sn = QtCore.QSocketNotifier(rsock.fileno(), QtCore.QSocketNotifier.Type.Read)

        @sn.activated.connect
        def _may_clear_sock():
            # Running a Python function on socket activation gives the interpreter a
            # chance to handle the signal in Python land.  We also need to drain the
            # socket with recv() to re-arm it, because it will be written to as part of
            # the wakeup.  (We need this in case set_wakeup_fd catches a signal other
            # than SIGINT and we shall continue waiting.)
            try:
                rsock.recv(1)
            except BlockingIOError:
                # This may occasionally fire too soon or more than once on Windows, so
                # be forgiving about reading an empty socket.
                pass

        return sn  # Actually keep the notifier alive.

    def handle_sigint():
        if hasattr(qapp_or_eventloop, 'closeAllWindows'):
            qapp_or_eventloop.closeAllWindows()
        qapp_or_eventloop.quit()

    return _allow_interrupt(prepare_notifier, handle_sigint)


PY312 = sys.version_info >= (3, 12)
