import itertools

from qtpy.QtCore import Qt, QModelIndex, QAbstractTableModel, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QMessageBox, QStyle

import numpy as np

from larray_editor.utils import (get_default_font,
                                 is_number_value, is_float_dtype, is_number_dtype,
                                 LinearGradient, logger, broadcast_get,
                                 format_exception, log_caller)

# TODO before first release:
# * add tests for reliable/"supported" adapters

# TODO before using the widget in other projects:
# * move ndigits/format detection to adapter
#   but the trick is to avoid using current column width and just
#   target a "reasonable number" of digits
# * update format on offset change (colwidth is updated but not format)
# * support thousand separators by default (see PROMES)
# * editing values on filtered arrays does not always work: when a filter
#   contains at least one dimension with *several* labels selected (but not
#   all), after editing a cell, the new value is visible only after the filter
#   is changed again. The cause of this is very ugly. There are actually
#   two distinct bugs at work here, usually hidden by the fact that filters
#   on larray usually return views all the way to the model. But when several
#   labels are selected, self.filtered_data in the LArray adapter is a *copy*
#   of the original data and not a view, so even if the model.reset()
#   (in EditObjecCommand) caused the model to re-ask the data from the
#   adapter (which it probably should but does NOT currently do -- it only
#   re-processes the data it already has -- which seems to work when the
#   data is a view), the adapter would still return the wrong data.
# * allow adapters to send more data than requested (unsure about this, maybe
#   implementing buffering in adapters is better)
# * implement the generic edit mechanism for the quickbar (to be able to edit
#   the original array while viewing a filtered array or, even, editing after
#   any other custom function, not just filtering)
#   - each adapter may provide a method:
#       can_edit_through_operation(op_name, args, kwargs)
#     unsure whether to use *args and **kwargs instead of args and kwargs
#   - if the adapter returns True for the above method (AbstractAdapter
#     must return False whatever the operation is), it must also implement a
#     transform_changes_through_inverse_of_operation(op_name, ..., changes)
#     method (whatever its final name). op could be '__getitem__' but also
#     'growth_rate', or whatever.
#   - the UI must clearly display whether an array is editable. Adding
#     "readonly" in the window title is a good start but unsure it is enough.
#   - I wonder if filtering could be done generically (if adapters implement
#     an helper function, we get filtering *with* editing passthrough)
#   - in a bluesky world, the inverse op could ask for options (e.g. editing a
#     summed cell can be dispersed on the individual unsummed values in several
#     ways: proportional to original value, constant fraction, ...)
# * adapters should provide a method to get position with axes area of each axis
#   and a method for the opposite (given an xy position, get the axis number).
#   The abstract adapter should provide a default implementation and possibly
#   the choice between several common options (larray style, pandas style, ...)
#   larray style is annoying for filters though, so I am unsure I want to
#   support it, even though it is the most compact and visually appealing.
#   Maybe using it when there is no need for a \ (no two *named* axes in the
#   same cell) is a good option???
# * do not convert to a single numpy array in Adapter.get_data_values_and_attributes
#   because it converts mixed string/number datasets to all strings, which
#   in turn breaks a lot of functionalities (alignment of numeric cells
#   is wrong, plots on numeric cells are wrong -- though this one suffers from
#   its own explicit conversion to numpy array --, etc.)
#   Ideally, we should support per-cell dtype, but if we support
#   dense arrays, per column dtype and per row dtype that would be already much
#   better than what we have now.
# * changing from an array to another is sometimes broken (new array not
#   displayed, old array axes still present)
#   - I *think* this only happens when an Exception is raised in the adapter
#     or arraymodel. These exceptions should not happen in the first place, but
#     the widget should handle them gracefully.
# * allow fixing frac_digits or scientific from the API (see PROMES, I think)
# * better readonly behavior from PROMES
# * include all other widgets from PROMES
# * allow having no filters nor gradient chooser (see PROMES)
# * massive cleanup (many methods would probably be better in either their
#   superclass or one of their subclasses)

# TODO post release (move to issues):
# * take decoration (sort arrow) into account to compute column widths
# * mouse selection on "edges" should move the buffer
#   (it scrolls the internal viewport but does not change the offset)

h_align_map = {
    'left': Qt.AlignLeft,
    'center': Qt.AlignHCenter,
    'right': Qt.AlignRight,
}
v_align_map = {
    'top': Qt.AlignTop,
    'center': Qt.AlignVCenter,
    'bottom': Qt.AlignBottom,
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

def homogenous_shape(seq) -> tuple:
    """
    Returns the shape (size of each dimension) of nested sequences.
    Checks that nested sequences are homogeneous; if not,
    treats children as scalars.
    """
    # we cannot rely on .shape for object arrays because they could
    # contain sequences themselves
    if isinstance(seq, np.ndarray) and not seq.dtype.kind == 'O':
        return seq.shape
    elif isinstance(seq, (list, tuple, np.ndarray)):
        parent_length = len(seq)
        if parent_length == 0:
            return (0,)
        elif parent_length == 1:
            return (parent_length,) + homogenous_shape(seq[0])
        res = [parent_length]
        child_shapes = [homogenous_shape(child)
                        for child in seq]
        # zip length will be determined by the shortest shape, which is
        # exactly what we need
        for depth_lengths in zip(*child_shapes[1:]):
            first_child_length = depth_lengths[0]
            if all(length == first_child_length
                   for length in depth_lengths[1:]):
                res.append(first_child_length)
        return tuple(res)
    else:
        return ()


def homogenous_ndim(seq) -> int:
    return len(homogenous_shape(seq))


def assert_at_least_2d_or_empty(seq):
    seq_shape = homogenous_shape(seq)
    assert len(seq_shape) >= 2 or 0 in seq_shape, (
        f"sequence:\n{seq}\nshould be >=2D or empty but has shape {seq_shape}")


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
    default_buffer_rows = 40
    default_buffer_cols = 40

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
        # FIXME: unused
        self.flags_defaults = Qt.NoItemFlags
        self.processed_flags_data = None #Qt.NoItemFlags
        self.bg_gradient = None
        self.default_v_align = [[Qt.AlignVCenter]]

    def set_adapter(self, adapter):
        self.adapter = adapter
        self.h_offset = 0
        self.v_offset = 0
        self.nrows = 0
        self.ncols = 0
        self._get_current_data()
        self.reset()

    def set_h_offset(self, offset):
        # TODO: when moving in one direction only, we should make sure to only request data we do not have already
        #       (if there is overlap between the old "window" and the new one).
        self.set_offset(self.v_offset, offset)

    def set_v_offset(self, offset):
        self.set_offset(offset, self.h_offset)

    def set_offset(self, v_offset, h_offset):
        # TODO: the implementation of this method should use set_bounds instead
        logger.debug(f"{self.__class__.__name__}.set_offset({v_offset=}, {h_offset=})")
        assert v_offset is not None and h_offset is not None
        assert v_offset >= 0 and h_offset >= 0
        self.v_offset = v_offset
        self.h_offset = h_offset
        old_shape = self.nrows, self.ncols
        self._get_current_data()
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

    # TODO: make this a private method (it is not called anymore but SHOULD be
    #       called (or inlined) in set_offset)
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
        self._get_current_data()
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

    def _format_value(self, args):
        fmt, value = args
        # using str(value) instead of '%s' % value makes it work for
        # tuple value
        # try:
        return fmt % value if is_number_value(value) else str(value)
        # except Exception as e:
        #     print("YAAAAAAAAAAAAAAA")
        #     return '<failed formatting value>'

    def _value_to_h_align(self, elem_value):
        if is_number_value(elem_value):
            return Qt.AlignRight
        else:
            return Qt.AlignLeft

    def _process_data(self):
        # None format => %user_format if number else %s

        # format per cell (in data) => decimal select will not work and that's fine
        # we could make decimal select change adapter-provided format per cell
        # which means the adapter can decide to ignore it or not
        # default adapter implementation should affect only numeric cells and use %s for non numeric
        raw_data = self.raw_data
        values = raw_data.get('values', [['']])
        assert_at_least_2d_or_empty(values)

        data_format = raw_data.get('data_format', [['%s']])
        assert_at_least_2d_or_empty(data_format)

        format_and_values = seq_zip_broadcast(data_format, values, ndim=2)
        formatted_values = map_nested_sequence(self._format_value,
                                               format_and_values,
                                               ndim=2)

        editable = raw_data.get('editable', [[False]])

        assert_at_least_2d_or_empty(editable)

        def editable_to_flags(elem_editable):
            editable_flag = Qt.ItemIsEditable if elem_editable else Qt.NoItemFlags
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | editable_flag

        # FIXME: use self.flags_defaults
        #     self.processed_flags_data = self.flags_defaults
        self.processed_flags_data = map_nested_sequence(editable_to_flags, editable, 2)
        self.processed_data = {
            Qt.DisplayRole: formatted_values,
            # XXX: maybe split this into font_name, font_size, font_flags, or make that data item a dict itself
            # Qt.FontRole: None,
            # Qt.ToolTipRole: None,
        }

        if 'h_align' in raw_data:
            h_align = map_nested_sequence(h_align_map.__getitem__, raw_data['h_align'], 2)
        else:
            h_align = map_nested_sequence(self._value_to_h_align, values, 2)
        if 'v_align' in raw_data:
            v_align = map_nested_sequence(v_align_map.__getitem__, raw_data['v_align'], 2)
        else:
            v_align = self.default_v_align
        self.processed_data[Qt.TextAlignmentRole] = [
            [int(ha | va) for ha, va in seq_zip_broadcast(ha_row, va_row)] for
            ha_row, va_row in seq_zip_broadcast(h_align, v_align)
        ]

        if 'bg_value' in raw_data and self.bg_gradient is not None:
            bg_value = raw_data['bg_value']
            # TODO: implement a way to specify bg_gradient per row or per column
            bg_color = self.bg_gradient[bg_value] if bg_value is not None else None
            self.processed_data[Qt.BackgroundColorRole] = bg_color
        if 'decoration' in raw_data:
            standardIcon = self.dialog.style().standardIcon
            def make_icon(decoration):
                return standardIcon(DECORATION_MAPPING[decoration]) if decoration else None
            decoration_data = map_nested_sequence(make_icon, raw_data['decoration'], 2)
            self.processed_data[Qt.DecorationRole] = decoration_data

    def _fetch_data(self, h_start, v_start, h_stop, v_stop):
        raise NotImplementedError()

    def _get_current_data(self):
        max_row, max_col = self.adapter.shape2d()

        # TODO: I don't think we should ever *ask* for more rows or columns
        #       than the default, but we should support *receiving* more,
        #       including before the current v_offset/h_offset.
        #       The only requirement should be that the asked for region
        #       is included in what we receive.
        rows_to_ask = max(self.nrows, self.default_buffer_rows)
        cols_to_ask = max(self.ncols, self.default_buffer_cols)
        h_stop = min(self.h_offset + cols_to_ask, max_col)
        v_stop = min(self.v_offset + rows_to_ask, max_row)
        # print(f"asking {rows_to_ask} rows / {cols_to_ask} columns of data ({self.__class__.__name__})")
        try:
            raw_data = self._fetch_data(self.h_offset, self.v_offset,
                                        h_stop, v_stop)
        except Exception as e:
            logger.error(f"could not fetch data from adapter:\n"
                         f"{''.join(format_exception(e))}")
            raw_data = np.array([[]])
        if not isinstance(raw_data, dict):
            raw_data = {'values': raw_data}
        self.raw_data = raw_data

        # XXX: currently this can be a view on the original data
        values = self.raw_data['values']
        assert_at_least_2d_or_empty(values)
        self.nrows = len(values)
        # FIXME: this is problematic for list of sequences
        first_row = values[0] if len(values) else []
        self.ncols = len(first_row) if isinstance(first_row, (tuple, list, np.ndarray)) else 1
        # print(f" > received {self.nrows} rows / {self.ncols} cols")
        if self.nrows > max_row:
            print(f"WARNING: received too many rows ({self.nrows} > {max_row})!")
        if self.ncols > max_col:
            print(f"WARNING: received too many columns ({self.ncols} > {max_col})!")

    def set_bg_gradient(self, bg_gradient):
        if bg_gradient is not None and not isinstance(bg_gradient, LinearGradient):
            raise ValueError("Expected None or LinearGradient instance for `bg_gradient` argument")
        self.bg_gradient = bg_gradient
        self.reset()

    def rowCount(self, parent=QModelIndex()):
        return self.nrows

    def columnCount(self, parent=QModelIndex()):
        return self.ncols

    # AFAICT, this is only used in the ArrayDelegate
    def get_value(self, index):
        return broadcast_get(self.raw_data['values'], index.row(), index.column())

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags

        row = index.row()
        column = index.column()
        if row >= self.nrows or column >= self.ncols:
            assert False, "should not have happened"
            return QAbstractTableModel.flags(self, index)
        return broadcast_get(self.processed_flags_data, row, column)

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
        log_caller()


DECORATION_MAPPING = {
    # QStyle.SP_TitleBarUnshadeButton
    # QStyle.SP_TitleBarShadeButton
    'arrow_down': QStyle.SP_ArrowDown,
    'arrow_up': QStyle.SP_ArrowUp,
}


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
        # TODO: move defaults to class attributes, not instances'
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
        self.flags_defaults = Qt.ItemIsEnabled

    def _fetch_data(self, h_start, v_start, h_stop, v_stop):
        axes_area = self.adapter.get_axes_area()
        # print(f"{axes_area=}")
        return axes_area

    def _value_to_h_align(self, elem_value):
        return Qt.AlignCenter

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
        self.flags_defaults = Qt.ItemIsEnabled

    def _value_to_h_align(self, elem_value):
        return Qt.AlignCenter

    # XXX: I wonder if we shouldn't return a 2D Numpy array of strings?
    # def get_values(self, left=0, top=0, right=None, bottom=None):
    #     if right is None:
    #         right = self.total_rows
    #     if bottom is None:
    #         bottom = self.total_cols
    #     values = [list(line[left:right]) for line in self._data[top:bottom]]
    #     return values


class VLabelsArrayModel(LabelsArrayModel):
    def _fetch_data(self, h_start, v_start, h_stop, v_stop):
        return self.adapter.get_vlabels(v_start, v_stop)


class HLabelsArrayModel(LabelsArrayModel):
    def _fetch_data(self, h_start, v_start, h_stop, v_stop):
        return self.adapter.get_hlabels(h_start, h_stop)


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


def seq_zip_broadcast(*seqs, ndim=1):
    """
    Zip sequences but broadcasting (repeating) s_len 1 sequences to the s_len
    of the longest sequence.

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
    ValueError: all sequences lengths must be 1 or the same:
    (['a1', 'a2'], ['b1', 'b2', 'b3'])
    >>> list(seq_zip_broadcast([[1], [2]], [[1, 2]], ndim=2))
    [[(1, 1), (1, 2)], [(2, 1), (2, 2)]]
    """
    if ndim == 1:
        assert len(seqs) > 0

        # "if s_len != 1" and "default=1" are necessary to support combining
        # an empty sequence with a length 1 sequence
        seq_lengths = [len(seq) for seq in seqs]
        max_length = max([s_len for s_len in seq_lengths if s_len != 1],
                         default=1)
        if not all(s_len in {1, max_length} for s_len in seq_lengths):
            raise ValueError(f"all sequences lengths must be 1 or the same:\n"
                             f"{seqs}")
        return zip(*(itertools.repeat(seq[0], max_length) if len(seq) == 1 else seq
                     for seq in seqs))
    else:
        assert ndim > 1
        broadcasted = seq_zip_broadcast(*seqs, ndim=1)
        return [list(seq_zip_broadcast(*seq, ndim=ndim - 1))
                for seq in broadcasted]


def map_nested_sequence(func, seq, ndim):
    """
    Apply a function to elements of a (nested) sequence at a specified depth.

    Parameters
    ----------
    func : callable
        Function to apply to elements at the target dimension level.
        Should accept a single argument and return a transformed value.
    seq : sequence
        The (potentially nested) sequence to process.
    ndim : int
        Target dimension depth. Must be >= 1.
        When ndim=1, applies func to each element of seq.
        When ndim>1, recursively processes nested sub-sequences.

    Returns
    -------
    list
        Returns a list with the same depth as the input with func applied to
        each element at the target depth.

    Examples
    --------
    >>> # 1D sequence
    >>> map_nested_sequence(lambda x: x * 2, [1, 2, 3], 1)
    [2, 4, 6]
    >>> # 2D sequence
    >>> map_nested_sequence(lambda x: x * 10, [[1, 2], [3, 4]], 2)
    [[10, 20], [30, 40]]
    >>> # Apply function to 2D sequence at depth 1
    >>> map_nested_sequence(str, [[1, 2], [3, 4]], 1)
    ['[1, 2]', '[3, 4]']
    """
    assert ndim >= 1
    if ndim == 1:
        return [func(elem) for elem in seq]
    else:
        return [map_nested_sequence(func, elem, ndim - 1)
                for elem in seq]


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
        default_font = get_default_font()
        self.role_defaults = {
            Qt.TextAlignmentRole: int(Qt.AlignRight | Qt.AlignVCenter),
            Qt.FontRole: default_font,
            # Qt.ToolTipRole:
        }

    def _fetch_data(self, h_start, v_start, h_stop, v_stop):
        return self.adapter.get_data_values_and_attributes(h_start, v_start,
                                                           h_stop, v_stop)

    # TODO: use ast.literal_eval instead of convert_value?
    # TODO: do this in the adapter
    def convert_value(self, value):
        """
        Parameters
        ----------
        value : str
        """
        # TODO: this assumes the adapter sends us a numpy array. Is it
        #       in the contract? I thought other sequence were accepted too?
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
        # TODO: do not do this, as it might change the dtype along the way. For example:
        #       >>> print(np.asarray([1, 3.0, "toto"]))
        #       ['1' '3.0' 'toto']
        values = np.asarray(values)
        # FIXME: for some adapters, we cannot rely on having a single dtype
        #        the dtype could be per-column, per-row, per-cell, or even, for some adapters
        #        (e.g. list), not fixed/changeable dynamically
        #        => we need to ask the adapter for the dtype
        #        => we need to know *here* which cells are impacted
        # TODO: maybe ask the adapter to convert_values instead (there should be some base
        #       functionality in the parent class though)
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

    # TODO: I wonder if set_values should not actually change the data (but QUndoCommand would make this weird).
    #       If we do this, we might need ArrayEdtiorWidget.paste and DataArrayModel.setData to call another method
    #       "queueValueChange" or something like that.
    #       In any case it must be absolutely clear from either the method name, an argument (eg. update_data=False)
    #       or from the class name that the data is not changed directly.
    def set_values(self, top, left, bottom, right, values):
        """
        This does NOT actually change any data directly. It will emit a signal
        that the data was changed, which is intercepted by the undo-redo system
        which creates a command to change the values, execute it and
        call .reset() on this model, which fetches and displays the new data.

        It is apparently NOT possible to add a QUndoCommand onto the QUndoStack
        without executing it.

        Parameters
        ----------
        top : int
           in global filtered coordinates
           (already includes v_offset but is not filter-aware)
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
        # paste should make sure this is the case
        assert values_height == 1 or values_height == selection_height
        assert values_width == 1 or values_width == selection_width

        # convert to local coordinates
        local_top = top - self.v_offset
        local_left = left - self.h_offset
        local_bottom = bottom - self.v_offset
        local_right = right - self.h_offset
        assert (local_top >= 0 and local_bottom >= 0 and
                local_left >= 0 and local_right >= 0)

        # compute changes dict
        changes = {}
        # requires numpy 1.10
        new_values = np.broadcast_to(values, (selection_height, selection_width))
        old_values = self.raw_data['values']
        for j in range(selection_height):
            for i in range(selection_width):
                old_value = old_values[local_top + j, local_left + i]
                new_value = new_values[j, i]
                if new_value != old_value:
                    changes[top + j, left + i] = (old_value, new_value)

        if len(changes) > 0:
            # changes take into account the viewport/offsets but not the filter
            # the array widget will use the adapter to translate those changes
            # to global changes then push them to the undo/redo stack, which
            # will execute them and that will actually modify the array
            self.newChanges.emit(changes)

        top_left = self.index(local_top, local_left)
        # -1 because Qt index end bounds are inclusive
        bottom_right = self.index(local_bottom - 1, local_right - 1)

        # emitting dataChanged only makes sense because a signal .emit call only returns when all its
        # slots have executed, so the newChanges signal emitted above has already triggered the whole
        # chain of code which effectively changes the data
        self.dataChanged.emit(top_left, bottom_right)
        return top_left, bottom_right

    def setData(self, index, value, role=Qt.EditRole):
        """Cell content change
        index is in local 2D coordinates
        """
        if not index.isValid():
            return False
        row, col = index.row(), index.column()
        row += self.v_offset
        col += self.h_offset
        result = self.set_values(row, col, row + 1, col + 1, value)
        return result is not None
