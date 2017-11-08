from __future__ import absolute_import, division, print_function

import numpy as np
from larray_editor.utils import (get_font, from_qvariant, to_qvariant, to_text_string,
                                 is_float, is_number, LinearGradient, SUPPORTED_FORMATS, scale_to_01range,
                                 Product)
from qtpy.QtCore import Qt, QModelIndex, QAbstractTableModel
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QMessageBox

LARGE_SIZE = 5e5
LARGE_NROWS = 1e5
LARGE_COLS = 60


class AbstractArrayModel(QAbstractTableModel):
    """Labels Table Model.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Widget.
    data : array-like, optional
        Input data.
    readonly : bool, optional
        If True, data cannot be changed. False by default.
    font : QFont, optional
        Font. Default is `Calibri` with size 11.
    """
    ROWS_TO_LOAD = 500
    COLS_TO_LOAD = 40

    def __init__(self, parent=None, data=None, readonly=False, font=None):
        QAbstractTableModel.__init__(self)

        self.dialog = parent
        self.readonly = readonly

        if font is None:
            font = get_font("arreditor")
        self.font = font

        self._data = None
        self.rows_loaded = 0
        self.cols_loaded = 0
        self.total_rows = 0
        self.total_cols = 0
        self.set_data(data)

    def _set_data(self, data, changes=None):
        raise NotImplementedError()

    def set_data(self, data, changes=None, **kwargs):
        self._set_data(data, changes, **kwargs)
        self.reset()

    def rowCount(self, parent=QModelIndex()):
        return self.rows_loaded

    def columnCount(self, parent=QModelIndex()):
        return self.cols_loaded

    def fetch_more_rows(self):
        if self.total_rows > self.rows_loaded:
            remainder = self.total_rows - self.rows_loaded
            items_to_fetch = min(remainder, self.ROWS_TO_LOAD)
            self.beginInsertRows(QModelIndex(), self.rows_loaded,
                                 self.rows_loaded + items_to_fetch - 1)
            self.rows_loaded += items_to_fetch
            self.endInsertRows()

    def fetch_more_columns(self):
        if self.total_cols > self.cols_loaded:
            remainder = self.total_cols - self.cols_loaded
            items_to_fetch = min(remainder, self.COLS_TO_LOAD)
            self.beginInsertColumns(QModelIndex(), self.cols_loaded,
                                    self.cols_loaded + items_to_fetch - 1)
            self.cols_loaded += items_to_fetch
            self.endInsertColumns()

    def get_value(self, index):
        raise NotImplementedError()

    def _compute_rows_cols_loaded(self):
        # Use paging when the total size, number of rows or number of
        # columns is too large
        size = self.total_rows * self.total_cols
        if size > LARGE_SIZE:
            self.rows_loaded = min(self.ROWS_TO_LOAD, self.total_rows)
            self.cols_loaded = min(self.COLS_TO_LOAD, self.total_cols)
        else:
            if self.total_rows > LARGE_NROWS:
                self.rows_loaded = self.ROWS_TO_LOAD
            else:
                self.rows_loaded = self.total_rows
            if self.total_cols > LARGE_COLS:
                self.cols_loaded = self.COLS_TO_LOAD
            else:
                self.cols_loaded = self.total_cols

    def flags(self, index):
        raise NotImplementedError()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        return to_qvariant()

    def data(self, index, role=Qt.DisplayRole):
        raise NotImplementedError()

    def reset(self):
        self.beginResetModel()
        self.endResetModel()


class LabelsArrayModel(AbstractArrayModel):
    """Labels Table Model.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Widget.
    data : nested list or tuple, optional
        Input data.
    readonly : bool, optional
        If True, data cannot be changed. False by default.
    font : QFont, optional
        Font. Default is `Calibri` with size 11.
    """
    def __init__(self, parent=None, data=None, readonly=False, font=None):
        AbstractArrayModel.__init__(self, parent, data, readonly, font)
        self.font.setBold(True)

    def _set_data(self, data, changes=None):
        if data is None:
            data = [[]]
        # TODO: use sequence instead
        if not isinstance(data, (list, tuple, Product)):
            QMessageBox.critical(self.dialog, "Error", "Expected list, tuple or Product")
            data = [[]]
        self._data = data
        self.total_rows = len(data[0])
        self.total_cols = len(data) if self.total_rows > 0 else 0
        self._compute_rows_cols_loaded()

    def flags(self, index):
        """Set editable flag"""
        return Qt.ItemIsEnabled

    def get_value(self, index):
        i = index.row()
        j = index.column()
        # we need to inverse column and row because of the way ylabels are generated
        return str(self._data[j][i])

    # XXX: I wonder if we shouldn't return a 2D Numpy array of strings?
    def get_values(self, left=0, top=0, right=None, bottom=None):
        if right is None:
            right = self.total_rows
        if bottom is None:
            bottom = self.total_cols
        values = [list(line[left:right]) for line in self._data[top:bottom]]
        return values

    def data(self, index, role=Qt.DisplayRole):
        # print('data', index.column(), index.row(), self.rowCount(), self.columnCount(), '\n', self._data)
        if not index.isValid():
            return to_qvariant()

        if role == Qt.TextAlignmentRole:
            return to_qvariant(int(Qt.AlignCenter | Qt.AlignVCenter))
        elif role == Qt.FontRole:
            return self.font
        elif role == Qt.BackgroundColorRole:
            color = QColor(Qt.lightGray)
            color.setAlphaF(.4)
            return color
        elif role == Qt.DisplayRole:
            value = self.get_value(index)
            return to_qvariant(value)
        elif role == Qt.ToolTipRole:
            return to_qvariant()
        else:
            return to_qvariant()


class DataArrayModel(AbstractArrayModel):
    """Data Table Model.

    Parameters
    ----------
    data : Numpy ndarray, optional
        Input 2D array.
    format : str, optional
        Indicates how data are represented in cells.
        By default, they are represented as floats with 3 decimal points.
    readonly : bool, optional
        If True, data cannot be changed. False by default.
    font : QFont, optional
        Font. Default is `Calibri` with size 11.
    parent : QWidget, optional
        Parent Widget.
    bg_gradient : LinearGradient, optional
        Background color gradient
    bg_value : Numpy ndarray, optional
        Background color value. Must have the shape as data
    minvalue : scalar
        Minimum value allowed.
    maxvalue : scalar
        Maximum value allowed.
    """

    ROWS_TO_LOAD = 500
    COLS_TO_LOAD = 40

    def __init__(self, parent=None, data=None, readonly=False, format="%.3f", font=None,
                 bg_gradient=None, bg_value=None, minvalue=None, maxvalue=None):
        AbstractArrayModel.__init__(self, parent, data, readonly, font)
        self._format = format

        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self._set_data(data)
        self._set_bg_gradient(bg_gradient)
        self._set_bg_value(bg_value)
        # XXX: unsure this is necessary at all in __init__
        self.reset()

    def get_format(self):
        """Return current format"""
        # Avoid accessing the private attribute _format from outside
        return self._format

    def get_data(self):
        """Return data"""
        return self._data

    def _set_data(self, data, changes=None, reset_minmax=True):
        if changes is None:
            changes = {}
        self.changes = changes

        # TODO: check that data respects minvalue/maxvalue
        if data is None:
            data = np.empty((0, 0), dtype=np.int8)
        if not (isinstance(data, np.ndarray) and data.ndim == 2):
            QMessageBox.critical(self.dialog, "Error", "Expect Numpy ndarray of 2 dimensions")
        self._data = data

        dtype = data.dtype
        if dtype.names is None:
            dtn = dtype.name
            if dtn not in SUPPORTED_FORMATS and not dtn.startswith('str') \
                    and not dtn.startswith('unicode'):
                QMessageBox.critical(self.dialog, "Error", "{} arrays are currently not supported".format(dtn))
                return
        # for complex numbers, shading will be based on absolute value
        # but for all other types it will be the real part
        # TODO: there are a lot more complex dtypes than this. Is there a way to get them all in one shot?
        if dtype in (np.complex64, np.complex128):
            self.color_func = np.abs
        else:
            self.color_func = None
        # --------------------------------------
        self.total_rows, self.total_cols = self._data.shape
        if reset_minmax:
            self.reset_minmax()
        self._compute_rows_cols_loaded()

    def reset_minmax(self):
        data = self.get_values()
        try:
            color_value = self.color_func(data) if self.color_func is not None else data
            # ignore nan, -inf, inf (setting them to 0 or to very large numbers is not an option)
            color_value = color_value[np.isfinite(color_value)]
            self.vmin = float(np.nanmin(color_value))
            self.vmax = float(np.nanmax(color_value))
            self.bgcolor_possible = True
        # ValueError for empty arrays, TypeError for object/string arrays
        except (TypeError, ValueError):
            self.vmin = None
            self.vmax = None
            self.bgcolor_possible = False

    def set_format(self, format):
        """Change display format"""
        self._format = format
        self.reset()

    def set_bg_gradient(self, bg_gradient):
        self._set_bg_gradient(bg_gradient)
        self.reset()

    def _set_bg_gradient(self, bg_gradient):
        if bg_gradient is not None and not isinstance(bg_gradient, LinearGradient):
            raise ValueError("Expected None or LinearGradient instance for `bg_gradient` argument")
        self.bg_gradient = bg_gradient

    def set_bg_value(self, bg_value):
        self._set_bg_value(bg_value)
        self.reset()

    def _set_bg_value(self, bg_value):
        if bg_value is not None and not (isinstance(bg_value, np.ndarray) and bg_value.shape == self._data.shape):
            raise ValueError("Expected None or 2D Numpy ndarray with shape {} for `bg_value` argument"
                             .format(self._data.shape))
        self.bg_value = bg_value

    def get_value(self, index):
        i, j = index.row(), index.column()
        return self.changes.get((i, j), self._data[i, j])

    def flags(self, index):
        """Set editable flag"""
        if not index.isValid():
            return Qt.ItemIsEnabled
        flags = QAbstractTableModel.flags(self, index)
        if not self.readonly:
            flags |= Qt.ItemIsEditable
        return Qt.ItemFlags(flags)

    def data(self, index, role=Qt.DisplayRole):
        """Cell content"""
        if not index.isValid():
            return to_qvariant()
        # if role == Qt.DecorationRole:
        #     return ima.icon('editcopy')
        # if role == Qt.DisplayRole:
        #     return ""

        if role == Qt.TextAlignmentRole:
            return to_qvariant(int(Qt.AlignRight | Qt.AlignVCenter))
        elif role == Qt.FontRole:
            return self.font

        value = self.get_value(index)
        if role == Qt.DisplayRole:
            if value is np.ma.masked:
                return ''
            # for headers
            elif isinstance(value, str) and not isinstance(value, np.str_):
                return value
            else:
                return to_qvariant(self._format % value)
        elif role == Qt.BackgroundColorRole:
            if self.bgcolor_possible and self.bg_gradient is not None and value is not np.ma.masked:
                if self.bg_value is None:
                    v = float(self.color_func(value) if self.color_func is not None else value)
                    v = scale_to_01range(v, self.vmin, self.vmax)
                else:
                    i, j = index.row(), index.column()
                    v = self.bg_value[i, j]
                return self.bg_gradient[v]
        # elif role == Qt.ToolTipRole:
        #     return to_qvariant("{}\n{}".format(repr(value),self.get_labels(index)))
        return to_qvariant()

    def get_values(self, left=0, top=0, right=None, bottom=None):
        width, height = self.total_rows, self.total_cols
        if right is None:
            right = width
        if bottom is None:
            bottom = height
        values = self._data[left:right, top:bottom].copy()
        # both versions get the same result, but depending on inputs, the
        # speed difference can be large.
        if values.size < len(self.changes):
            for i in range(left, right):
                for j in range(top, bottom):
                    pos = i, j
                    if pos in self.changes:
                        values[i - left, j - top] = self.changes[pos]
        else:
            for (i, j), value in self.changes.items():
                if left <= i < right and top <= j < bottom:
                    values[i - left, j - top] = value
        return values

    def convert_value(self, value):
        """
        Parameters
        ----------
        value : str
        """
        dtype = self._data.dtype
        if dtype.name == "bool":
            try:
                return bool(float(value))
            except ValueError:
                return value.lower() == "true"
        elif dtype.name.startswith("string"):
            return str(value)
        elif dtype.name.startswith("unicode"):
            return to_text_string(value)
        elif is_float(dtype):
            return float(value)
        elif is_number(dtype):
            return int(value)
        else:
            return complex(value)

    def convert_values(self, values):
        values = np.asarray(values)
        res = np.empty_like(values, dtype=self._data.dtype)
        try:
            # TODO: use array/vectorized conversion functions (but watch out
            # for bool)
            # new_data = str_array.astype(data.dtype)
            for i, v in enumerate(values.flat):
                res.flat[i] = self.convert_value(v)
        except ValueError as e:
            QMessageBox.critical(self.dialog, "Error", "Value error: %s" % str(e))
            return None
        except OverflowError as e:
            QMessageBox.critical(self.dialog, "Error", "Overflow error: %s" % e.message)
            return None
        return res

    def set_values(self, left, top, right, bottom, values):
        """
        Parameters
        ----------
        left : int
        top : int
        right : int
            exclusive
        bottom : int
            exclusive
        values : ndarray
            must not be of the correct type

        Returns
        -------
        tuple of QModelIndex or None
            actual bounds (end bound is inclusive) if update was successful,
            None otherwise
        """
        values = self.convert_values(values)
        if values is None:
            return
        values = np.atleast_2d(values)
        vshape = values.shape
        vwidth, vheight = vshape
        width, height = right - left, bottom - top
        assert vwidth == 1 or vwidth == width
        assert vheight == 1 or vheight == height

        # Add change to self.changes
        # requires numpy 1.10
        newvalues = np.broadcast_to(values, (width, height))
        oldvalues = np.empty_like(newvalues)
        for i in range(width):
            for j in range(height):
                pos = left + i, top + j
                old_value = self.changes.get(pos, self._data[pos])
                oldvalues[i, j] = old_value
                val = newvalues[i, j]
                if val != old_value:
                    self.changes[pos] = val

        # Update vmin/vmax if necessary
        if self.vmin is not None and self.vmax is not None:
            colorval = self.color_func(values) if self.color_func is not None else values
            old_colorval = self.color_func(oldvalues) if self.color_func is not None else oldvalues
            if np.any(((old_colorval == self.vmax) & (colorval < self.vmax)) |
                      ((old_colorval == self.vmin) & (colorval > self.vmin))):
                self.reset_minmax()
            # this is faster, when the condition is False (which should be most of the cases) than computing
            # subset_max and checking if subset_max > self.vmax
            if np.any(colorval > self.vmax):
                self.vmax = float(np.nanmax(colorval))
            if np.any(colorval < self.vmin):
                self.vmin = float(np.nanmin(colorval))

        top_left = self.index(left, top)
        # -1 because Qt index end bounds are inclusive
        bottom_right = self.index(right - 1, bottom - 1)
        self.dataChanged.emit(top_left, bottom_right)
        return top_left, bottom_right

    def setData(self, index, value, role=Qt.EditRole):
        """Cell content change"""
        if not index.isValid() or self.readonly:
            return False
        i, j = index.row(), index.column()
        result = self.set_values(i, j, i + 1, j + 1, from_qvariant(value, str))
        return result is not None

    def reset(self):
        self.beginResetModel()
        self.endResetModel()
