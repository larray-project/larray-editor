import itertools
from os.path import basename
import logging
from inspect import stack

from qtpy.QtCore import Qt, QModelIndex, QAbstractTableModel, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QMessageBox

import numpy as np

from larray_editor.utils import (get_default_font,
                                 is_number_value, is_float_dtype, is_number_dtype,
                                 LinearGradient, Product, logger, broadcast_get)

# FIXME:
# * scrolling via the scrollbar cannot reach some columns (if the viewport is < buffer) because
#   it does not allow scrolling the internal
# * selection vs moving the buffer offset

# * mouse selection on "edges" does not move the buffer
# * changed column width vs moving the buffer offset
# * scrolling on filtered arrays gives wrong results
# * editing values on filtered array does not work
# * cell colors vs moving the buffer offset
# * changing from an array to another is sometimes broken (new array not displayed, old array axes
#   still present)
# * paste does not work (copy works fine)
# * massive cleanup (many methods would probably be better in either their superclass
#   or one of their subclasses)
# * update initial sizes

role_map = {
    'data': Qt.DisplayRole,
    # XXX: still unsure who should handle bg_value -> color conversion
    # maybe we could offer both (exclusive): "bg_value" => conversion done in model while
    # bg_color would provide direct rgb colors (in Qt object or some other form to be decided?)
    # * I would like to avoid using Qt objects in the adapter
    # * an adapter could want to use several gradient for different regions
    'bg_value': Qt.BackgroundColorRole,
    'text_align': Qt.TextAlignmentRole,
    # XXX: maybe split this into font_name, font_size, font_flags, or make that data item a dict itself
    'font': Qt.FontRole,
    'tooltip': Qt.ToolTipRole,
}

#    h_offset
#    --------
#      | |
#      v v
#      |----------------------| <-|
#      |    total data        |   | v_offset
#      | |------------------| | <-|
#      | |  data in model   | |
#      | | |--------------| | |
#      | | | visible area | | |
#      | | |--------------| | |
#      | |------------------| |
#      |----------------------|


class AbstractArrayModel(QAbstractTableModel):
    """Labels Table Model.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Widget.
    readonly : bool, optional
        If True, data cannot be changed. False by default.
    font : QFont, optional
        Font. Default is `Calibri` with size 11.
    """
    default_buffer_rows = 30
    default_buffer_cols = 10

    def __init__(self, parent=None, adapter=None):
        QAbstractTableModel.__init__(self)

        self.dialog = parent
        self.adapter = adapter

        self.h_offset = 0
        self.v_offset = 0
        self.nrows = 0
        self.ncols = 0

        self.raw_data = {}
        self.processed_data = {}
        self.role_defaults = {}

    # def _set_data(self, data):
    #     raise NotImplementedError()

    def set_adapter(self, adapter):
        self.adapter = adapter
        self.h_offset = 0
        self.v_offset = 0
        self.nrows = 0
        self.ncols = 0
        self._get_data()
        self.reset()

    def set_h_offset(self, offset):
        # TODO: when moving in one direction only, we should make sure to only request data we do not have already
        #       (if there is overlap between the old "window" and the new one).
        self.set_offset(self.v_offset, offset)

    def set_v_offset(self, offset):
        self.set_offset(offset, self.h_offset)

    def set_offset(self, v_offset, h_offset):
        # TODO: the implementation of this method should use set_bounds instead
        print("set v/h offset to", v_offset, h_offset)
        assert v_offset is not None and h_offset is not None
        assert v_offset >= 0 and h_offset >= 0
        self.v_offset = v_offset
        self.h_offset = h_offset
        old_shape = self.nrows, self.ncols
        self._get_data()
        self._process_data()
        new_shape = self.nrows, self.ncols
        if new_shape != old_shape:
            self.reset()
        else:
            top_left = self.index(0, 0)
            # -1 because Qt index end bounds are inclusive
            bottom_right = self.index(self.nrows - 1, self.ncols - 1)
            self.dataChanged.emit(top_left, bottom_right)

    def _begin_insert_remove(self, action, target, parent, start, stop):
        if start >= stop:
            return False
        funcs = {
            ('remove', 'rows'): self.beginRemoveRows,
            ('insert', 'rows'): self.beginInsertRows,
            ('remove', 'columns'): self.beginRemoveColumns,
            ('insert', 'columns'): self.beginInsertColumns,
        }
        funcs[action, target](parent, start, stop - 1)
        return True

    def _end_insert_remove(self, action, target):
        funcs = {
            ('remove', 'rows'): self.endRemoveRows,
            ('insert', 'rows'): self.endInsertRows,
            ('remove', 'columns'): self.endRemoveColumns,
            ('insert', 'columns'): self.endInsertColumns,
        }
        funcs[action, target]()

    # FIXME: review all API methods and be consistent in argument order: h, v or v, h.
    #        I think Qt always uses v, h but we sometime use h, v

    # TODO: make this a private method (it is not called anymore but SHOULD be called (or inlined) in set_offset)
    def set_bounds(self, v_start=None, h_start=None, v_stop=None, h_stop=None):
        """stop bounds are *exclusive*
        any None is replaced by its previous value"""

        oldvstart, oldhstart = self.v_offset, self.h_offset
        oldvstop, oldhstop = oldvstart + self.nrows, oldhstart + self.ncols
        newvstart, newhstart, newvstop, newhstop = v_start, h_start, v_stop, h_stop
        print("set bounds", v_start, h_start, v_stop, h_stop)
        if newvstart is None:
            newvstart = oldvstart
        if newhstart is None:
            newhstart = oldhstart
        if newvstop is None:
            newvstop = oldvstop
        if newhstop is None:
            newhstop = oldhstop

        nrows = newvstop - newvstart
        ncols = newhstop - newhstart
        # new_shape = nrows, ncols

        # if new_shape != old_shape:
        #     self.reset()
        # else:

        # we could generalize this to allow moving the "viewport" and shrinking/enlarging it at the same time
        # but this is a BAD idea as we should very rarely shrink/enlarge the buffer

        # assert we_are_enlarging or shrinking in one direction only
        # we have 9 cases total: same shape or 4 cases for each direction: enlarging|shrinking * moving start|stop
        # ensure we have some overlap between old and new

        #                                              ENLARGE
        #                                              -------

        #  start  stop             start  stop                    start  stop       start  stop
        #  |---old---|             |---old---|                    |---old---|       |---old---|
        #  v         v             v         v                    v         v       v         v
        # ----------------- OR ---------------- OR --------------------------- OR  --------------------------
        #    ^           ^      ^           ^       ^           ^                              ^           ^
        #    |----new----|      |----new----|       |----new----|                              |----new----|
        #    start    stop      start    stop       start    stop                              start    stop

        #                                              SHRINK
        #                                              ------

        #  start    stop       start    stop                 start    stop       start    stop
        #  |----old----|       |----old----|                 |----old----|       |----old----|
        #  v           v       v           v                 v           v       v           v
        # --------------- OR ---------------- OR -------------------------- OR  --------------------------
        #   ^         ^       ^         ^         ^         ^                                 ^         ^
        #   |---new---|       |---new---|         |---new---|                                 |---new---|
        #   start  stop       start  stop         start  stop                                 start  stop

        parent = QModelIndex()

        end_todo = {}
        for action in ('remove', 'insert'):
            for target in ('rows', 'columns'):
                end_todo[action, target] = False

        target = 'rows'
        oldstart, oldstop, newstart, newstop = oldvstart, oldvstop, newvstart, newvstop

        # remove oldstart:newstart
        end_todo['remove', target] |= self._begin_insert_remove('remove', target, parent,
                                                                oldstart, min(newstart, oldstop))
        # remove newstop:oldstop
        end_todo['remove', target] |= self._begin_insert_remove('remove', target, parent,
                                                                max(newstop, oldstart), oldstop)
        # insert newstart:oldstart
        end_todo['insert', target] |= self._begin_insert_remove('insert', target, parent,
                                                                newstart, min(oldstart, newvstop))
        # insert oldstop:newstop
        end_todo['insert', target] |= self._begin_insert_remove('insert', target, parent,
                                                                max(oldstop, newstart), newstop)

        target = 'columns'
        oldstart, oldstop, newstart, newstop = oldhstart, oldhstop, newhstart, newhstop

        # remove oldstart:newstart
        end_todo['remove', target] |= self._begin_insert_remove('remove', target, parent,
                                                                oldstart, min(newstart, oldstop))
        # remove newstop:oldstop
        end_todo['remove', target] |= self._begin_insert_remove('remove', target, parent,
                                                                max(newstop, oldstart), oldstop)
        # insert newstart:oldstart
        end_todo['insert', target] |= self._begin_insert_remove('insert', target, parent,
                                                                newstart, min(oldstart, newvstop))
        # insert oldstop:newstop
        end_todo['insert', target] |= self._begin_insert_remove('insert', target, parent,
                                                                max(oldstop, newstart), newstop)

        assert newvstart is not None and newhstart is not None
        self.v_offset, self.h_offset = newvstart, newhstart
        self.nrows, self.ncols = nrows, ncols

        # TODO: we should only get data we do not have yet
        self._get_data()
        # TODO: we should only process data we do not have yet
        self._process_data()

        for action in ('remove', 'insert'):
            for target in ('rows', 'columns'):
                if end_todo[action, target]:
                    self._end_insert_remove(action, target)

        # removed_rows_start, remove_rows_stop = ... # correspond to old_rows - nrows
        # changed_rows_start, changed_rows_stop = ... # other rows
        # if v_stop > oldvstop and v_start < oldvstop:  # 10 < 11 => 10.. & ..10 => intersection = 10
        #     insertRows
        # elif v_start < oldvstart and v_stop > oldvstart:  # 11 > 10 => ..10 & 10.. => intersection = 10
        #     insertRows
        # elif v_stop < oldvstop and v_stop > oldvstart:  # 11 > 10 => ..10 & 10.. => intersection = 10
        #     removeRows
        # elif v_start < oldvstart and v_stop > oldvstart:  # 11 > 10 => ..10 & 10.. => intersection = 10
        #     removeRows
        # if new_shape == old_shape:
        #     # FIXME: this method will never be called for this case, unless I suppress set_offset
        #     top_left = self.index(0, 0)
        #     # -1 because Qt index end bounds are inclusive
        #     bottom_right = self.index(nrows - 1, ncols - 1)
        #     self.dataChanged.emit(top_left, bottom_right)
        # elif nrows == old_rows and ncols > old_cols:
        #     if ...:
        #         pass
        #     else:
        #         pass
        # elif nrows == old_rows and ncols < old_cols:
        #     if ...:
        #         pass
        #     else:
        #         ...
        # elif nrows == old_rows and ncols < old_cols:
        #     self.beginRemoveRows(parent, first, last)
        #     self.beginInsertRows(QModelIndex(), self.nrows, self.nrows + items_to_fetch - 1)
        #     self.nrows += items_to_fetch
        #     self._get_data()
        #     self._process_data()
        #     self.endInsertRows()
        #     top_left = self.index(0, 0)
        #     # -1 because Qt index end bounds are inclusive
        #     bottom_right = self.index(nrows - 1, ncols - 1)
        #     self.dataChanged.emit(top_left, bottom_right)

    def _process_data(self):
        pass

    def _get_data(self):
        raise NotImplementedError()

    def rowCount(self, parent=QModelIndex()):
        return self.nrows

    def columnCount(self, parent=QModelIndex()):
        return self.ncols

    # FIXME: either move this to DataArrayModel (raise NotImplementedError here) or use raw_data & process_data
    #        for other models too.
    # AFAICT, this is only used in the ArrayDelegate
    def get_value(self, index):
        return broadcast_get(self.raw_data['values'], index.row(), index.column())

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled
        flags = self.data(index, 'flags')
        if flags is not None:
            return flags
        else:
            return QAbstractTableModel.flags(self, index)
            # return Qt.ItemIsEnabled

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        role_map = self.processed_data
        if role in role_map:
            role_data = role_map[role]
        else:
            role_data = self.role_defaults.get(role)

        row = index.row()
        column = index.column()
        if row >= self.nrows or column >= self.ncols:
            return None
        return broadcast_get(role_data, row, column)
        # res = broadcast_get_index(role_data, index)
        # if role == Qt.DisplayRole:
        #     print("data", index.row(), index.column(), "=>", res)
        # return res

    def reset(self):
        self.beginResetModel()
        self._process_data()
        self.endResetModel()
        if logger.isEnabledFor(logging.DEBUG):
            caller = stack()[1]
            logger.debug(f"model {self.__class__} has been reset after call of {caller.function} from module "
                         f"{basename(caller.filename)} at line {caller.lineno}")


class AxesArrayModel(AbstractArrayModel):
    """Axes Table Model.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Widget.
    readonly : bool, optional
        If True, data cannot be changed. False by default.
    font : QFont, optional
        Font. Default is `Calibri` with size 11.
    """
    def __init__(self, parent=None, adapter=None):
        AbstractArrayModel.__init__(self, parent, adapter)
        default_font = get_default_font()
        default_font.setBold(True)
        default_background = QColor(Qt.lightGray)
        default_background.setAlphaF(.4)
        self.role_defaults = {
            Qt.TextAlignmentRole: int(Qt.AlignCenter | Qt.AlignVCenter),
            Qt.FontRole: default_font,
            Qt.BackgroundColorRole: default_background,
            # Qt.DisplayRole: '',
            # Qt.ToolTipRole:
        }

    def _get_data(self):
        names = self.adapter.get_axes_area()
        self.processed_data = {
            Qt.DisplayRole: names
        }
        self.nrows = len(names)
        self.ncols = len(names[0])

    # def get_values(self, left=0, right=None):
    #     if right is None:
    #         right = self.total_cols
    #     values = self._data[left:right]
    #     return values


class LabelsArrayModel(AbstractArrayModel):
    """Labels Table Model.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Widget.
    readonly : bool, optional
        If True, data cannot be changed. False by default.
    font : QFont, optional
        Font. Default is `Calibri` with size 11.
    """
    def __init__(self, parent=None, adapter=None):
        AbstractArrayModel.__init__(self, parent, adapter)
        default_font = get_default_font()
        default_font.setBold(True)
        default_background = QColor(Qt.lightGray)
        default_background.setAlphaF(.4)
        self.role_defaults = {
            Qt.TextAlignmentRole: int(Qt.AlignCenter | Qt.AlignVCenter),
            Qt.FontRole: default_font,
            Qt.BackgroundColorRole: default_background,
            # Qt.ToolTipRole:
        }

    # XXX: I wonder if we shouldn't return a 2D Numpy array of strings?
    # def get_values(self, left=0, top=0, right=None, bottom=None):
    #     if right is None:
    #         right = self.total_rows
    #     if bottom is None:
    #         bottom = self.total_cols
    #     values = [list(line[left:right]) for line in self._data[top:bottom]]
    #     return values


class VLabelsArrayModel(LabelsArrayModel):
    def _get_data(self):
        max_row, max_col = self.adapter.shape2d()

        rows_to_ask = max(self.nrows, self.default_buffer_rows)
        v_stop = min(self.v_offset + rows_to_ask, max_row)

        print("asking", rows_to_ask, "vlabels")
        # TODO: the adapter should have the possibility to return a dict, like for DataArrayModel.
        #
        labels = self.adapter.get_vlabels(self.v_offset, v_stop)
        print(f" > received {len(labels)}")
        self.processed_data = {
            Qt.DisplayRole: [[str(l) for l in row] for row in labels],
            'flags': Qt.ItemIsEnabled,
        }
        self.nrows = len(labels)
        self.ncols = len(labels[0]) if len(labels) else 0


class HLabelsArrayModel(LabelsArrayModel):
    def _get_data(self):
        max_row, max_col = self.adapter.shape2d()

        cols_to_ask = max(self.ncols, self.default_buffer_cols)
        h_stop = min(self.h_offset + cols_to_ask, max_col)

        print("asking", cols_to_ask, "hlabels")
        labels = self.adapter.get_hlabels(self.h_offset, h_stop)
        print(f" > received {len(labels)}")
        self.processed_data = {
            Qt.DisplayRole: [[str(l) for l in row] for row in labels],
            # TODO: move this to role_defaults
            'flags': Qt.ItemIsEnabled,
        }
        self.nrows = len(labels)
        self.ncols = len(labels[0]) if len(labels) else 0


def seq_broadcast(*seqs):
    """
    Examples
    --------
    >>> seq_broadcast(["a"], ["b1", "b2"])
    (['a', 'a'], ['b1', 'b2'])
    >>> seq_broadcast(["a1", "a2"], ["b"])
    (['a1', 'a2'], ['b', 'b'])
    >>> seq_broadcast(["a1", "a2"], ["b1", "b2"])
    (['a1', 'a2'], ['b1', 'b2'])
    >>> seq_broadcast(["a1", "a2"], ["b1", "b2", "b3"])
    Traceback (most recent call last):
    ...
    ValueError: all sequences lengths must be 1 or the same
    """
    seqs = [seq if isinstance(seq, (tuple, list)) else [seq]
            for seq in seqs]
    assert all(hasattr(seq, '__getitem__') for seq in seqs)
    length = max(len(seq) for seq in seqs)
    if not all(len(seq) == 1 or len(seq) == length for seq in seqs):
        raise ValueError("all sequences lengths must be 1 or the same")
    return tuple(seq * length if len(seq) == 1 else seq
                 for seq in seqs)


def seq_zip_broadcast(*seqs):
    """
    Zip sequences but broadcasting (repeating) length 1 sequences to the length of the
    longest sequence.

    Examples
    --------
    >>> list(seq_zip_broadcast(["a"], ["b1", "b2"]))
    [('a', 'b1'), ('a', 'b2')]
    >>> list(seq_zip_broadcast(["a1", "a2"], ["b"]))
    [('a1', 'b'), ('a2', 'b')]
    >>> list(seq_zip_broadcast(["a1", "a2"], ["b1", "b2"]))
    [('a1', 'b1'), ('a2', 'b2')]
    >>> list(seq_zip_broadcast(["a1", "a2"], ["b1", "b2", "b3"]))
    Traceback (most recent call last):
    ...
    ValueError: all sequences lengths must be 1 or the same
    """
    # this is the tricky part
    # TODO: accept Sequence too
    sequence = (tuple, list, np.ndarray)
    seqs = [seq if isinstance(seq, sequence) or
            (isinstance(seq, np.void) and seq.dtype.names is not None)
            else [seq]
            for seq in seqs]
    # assert all(hasattr(seq, '__getitem__') for seq in seqs)
    length = 1 if all(len(seq) == 1 for seq in seqs) else max(len(seq) for seq in seqs if len(seq) != 1)
    if not all(len(seq) == 1 or len(seq) == length for seq in seqs):
        raise ValueError(f"all sequences lengths must be 1 or the same: {seqs}")
    return zip(*(itertools.repeat(seq[0], length) if len(seq) == 1 else seq
                 for seq in seqs))
    # return zip(*(seq * length if len(seq) == 1 else seq
    #              for seq in seqs))


def map_nested_sequence(seq, func):
    sequence = (tuple, list, np.ndarray)
    if isinstance(seq, sequence):
        return [map_nested_sequence(elem, func) for elem in seq]
    else:
        return func(seq)


class DataArrayModel(AbstractArrayModel):
    """Data Table Model.

    Parameters
    ----------
    parent : QWidget, optional
        Parent Widget.
    FIXME: update this
    readonly : bool, optional
        If True, data cannot be changed. False by default.
    format : str, optional
        Indicates how data is represented in cells.
        By default, they are represented as floats with 3 decimal points.
    font : QFont, optional
        Font. Default is `Calibri` with size 11.
    """

    newChanges = Signal(dict)

    def __init__(self, parent=None, adapter=None):
        # readonly=False, format="%.3f", font=None):
        AbstractArrayModel.__init__(self, parent, adapter)
        # self._default_format = '%.3f'

        default_font = get_default_font()
        self.role_defaults = {
            Qt.TextAlignmentRole: int(Qt.AlignRight | Qt.AlignVCenter),
            Qt.FontRole: default_font,
            # Qt.ToolTipRole:
        }
        self.bg_gradient = None

    def _get_data(self):
        max_row, max_col = self.adapter.shape2d()

        rows_to_ask = max(self.nrows, self.default_buffer_rows)
        cols_to_ask = max(self.ncols, self.default_buffer_cols)
        h_stop = min(self.h_offset + cols_to_ask, max_col)
        v_stop = min(self.v_offset + rows_to_ask, max_row)
        print(f"asking {rows_to_ask} rows / {cols_to_ask} columns of data")
        self.raw_data = self.adapter.get_data(self.h_offset, self.v_offset, h_stop, v_stop)
        # XXX: currently this can be a view on the original data
        values = self.raw_data['values']
        self.nrows = len(values)
        # FIXME: this is problematic for list of sequences
        first_row = values[0] if len(values) else []
        self.ncols = len(first_row) if isinstance(first_row, (tuple, list, np.ndarray)) else 1
        print(f" > received {self.nrows} rows / {self.ncols} cols")
        if self.nrows > max_row:
            print(f"WARNING: received too many rows ({self.nrows} > {max_row})!")
        if self.ncols > max_col:
            print(f"WARNING: received too many columns ({self.ncols} > {max_col})!")

    def _process_data(self):
        # None format => %user_format if number else %s

        # format per cell (in data) => decimal select will not work and that's fine
        # we could make decimal select change adapter-provided format per cell
        # which means the adapter can decide to ignore it or not
        # default adapter implementation should affect only numeric cells and use %s for non numeric
        values = self.raw_data.get('values', '')

        if 'format_func' in self.raw_data:
            format_func = self.raw_data['format_func']
        else:
            def format_func(fmt, value):
                fmt = fmt if is_number_value(value) else '%s'
                return fmt % value

        # format_func = self.raw_data.get('format_func', [default_format])
        data_format = self.raw_data.get('data_format', '%s')

        formatted_values = [[func(fmt, value) for func, fmt, value in seq_zip_broadcast(rowfunc, rowfmt, rowvalue)]
                            for rowfunc, rowfmt, rowvalue in seq_zip_broadcast(format_func, data_format, values)]

        bg_value = self.raw_data.get('bg_value')
        # TODO: implement a way to specify bg_gradient per row or per column
        bg_color = self.bg_gradient[bg_value] if bg_value is not None else None

        editable = self.raw_data.get('editable')

        def editable_to_flags(elem_editable):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | (Qt.ItemIsEditable if elem_editable else 0)

        flags = map_nested_sequence(editable, editable_to_flags)
        self.processed_data = {
            Qt.DisplayRole: formatted_values,
            Qt.BackgroundColorRole: bg_color,
            # TODO: store flags in a separate dict?
            # TODO: move this to role_defaults
            'flags': flags,
            # XXX: maybe split this into font_name, font_size, font_flags, or make that data item a dict itself
            # Qt.FontRole: None,
            # Qt.ToolTipRole: None,
        }

    def set_bg_gradient(self, bg_gradient, reset=True):
        if bg_gradient is not None and not isinstance(bg_gradient, LinearGradient):
            raise ValueError("Expected None or LinearGradient instance for `bg_gradient` argument")
        self.bg_gradient = bg_gradient
        if reset:
            self.reset()

    # TODO: use ast.literal_eval instead of convert_value?
    def convert_value(self, value):
        """
        Parameters
        ----------
        value : str
        """
        dtype = self.raw_data['values'].dtype
        if dtype.name == "bool":
            try:
                return bool(float(value))
            except ValueError:
                return value.lower() == "true"
        elif dtype.name.startswith("string") or dtype.name.startswith("unicode"):
            return str(value)
        elif is_float_dtype(dtype):
            return float(value)
        elif is_number_dtype(dtype):
            return int(value)
        else:
            return complex(value)

    def convert_values(self, values):
        values = np.asarray(values)
        # FIXME: for some adapters, we cannot rely on having a single dtype
        #        the dtype could be per-column, per-row, per-cell, or even, for some adapters
        #        (e.g. list), not fixed/changeable dynamically
        dtype = self.raw_data['values'].dtype
        res = np.empty_like(values, dtype=dtype)
        try:
            # TODO: use array/vectorized conversion functions (but watch out
            # for bool)
            # new_data = str_array.astype(data.dtype)
            # TODO: do this in two steps. Get convertion_func for the dtype then call it
            for i, v in enumerate(values.flat):
                res.flat[i] = self.convert_value(v)
        except ValueError as e:
            QMessageBox.critical(self.dialog, "Error", f"Value error: {str(e)}")
            return None
        except OverflowError as e:
            QMessageBox.critical(self.dialog, "Error", f"Overflow error: {e.message}")
            return None
        return res

    # TODO: I wonder if set_values should not actually change the data. In that case, ArrayEdtiorWidget.paste
    # and DataArrayModel.setData should call another method "queueValueChange" or something like that. In any case
    # it must be absolutely clear from either the method name, an argument (eg. update_data=False) or from the
    # class name that the data is not changed directly.
    # I am also unsure how this all thing will interact with the big adapter/model refactor in the buffer branch.
    def set_values(self, top, left, bottom, right, values):
        """
        This does NOT actually change any data directly. It will emit a signal that the data was changed,
        which is intercepted by the undo-redo system which creates a command to change the values, execute it and
        call .reset() on this model, which fetches and displays the new data. It is apparently NOT possible to add a
        QUndoCommand onto the QUndoStack without executing it.

        To add to the strangeness, this method updates self.vmin and self.vmax immediately, which leads to very odd
        results (the color is updated but not the value) if one forgets to connect the newChanges signal to the
        undo-redo system.

        Parameters
        ----------
        top : int
        left : int
        bottom : int
            exclusive
        right : int
            exclusive
        values : ndarray
            may be of incorrect type

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
        values_height, values_width = values.shape
        selection_height, selection_width = bottom - top, right - left
        assert values_height == 1 or values_height == selection_height
        assert values_width == 1 or values_width == selection_width

        # compute changes dict
        changes = {}
        # requires numpy 1.10
        new_values = np.broadcast_to(values, (selection_height, selection_width))
        old_values = self.raw_data['values']
        for j in range(selection_height):
            for i in range(selection_width):
                old_value = old_values[top + j, left + i]
                new_value = new_values[j, i]
                if new_value != old_value:
                    changes[top + j + self.v_offset, left + i + self.h_offset] = (old_value, new_value)

        if len(changes) > 0:
            # the array widget will use the adapter to translate those changes to global changes then push them to
            # the undo/redo stack, which will execute them and that will actually modify the array
            self.newChanges.emit(changes)

        top_left = self.index(top, left)
        # -1 because Qt index end bounds are inclusive
        bottom_right = self.index(bottom - 1, right - 1)

        # emitting dataChanged only makes sense because a signal .emit call only returns when all its
        # slots have executed, so the newChanges signal emitted above has already triggered the whole
        # chain of code which effectively changes the data
        self.dataChanged.emit(top_left, bottom_right)
        return top_left, bottom_right

    def setData(self, index, value, role=Qt.EditRole):
        """Cell content change"""
        # FIXME: this is too late to check for readonly, the editor, should not be created by the ArrayDelegate
        #        in that case.
        if not index.isValid():
            return False
        row, col = index.row(), index.column()
        result = self.set_values(row, col, row + 1, col + 1, value)
        return result is not None
