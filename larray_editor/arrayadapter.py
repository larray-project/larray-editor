# TODO (long term): add support for streaming data source. In that case, the behavior should be what we had before
#                   (when we scroll, it requests more data and the total length of the scrollbar is updated)
#
# TODO (even longer term): add support for streaming data source with a limit (ie keep last N entries)
#
# TODO: add support for "progressive" data sources, e.g. pandas SAS reader
#       (from pandas.io.sas.sas7bdat import SAS7BDATReader), which can read by chunks but cannot
#       read a particular offset. It would be crazy to re-read the whole thing up to the requested
#       data each time, but caching the whole file in memory probably isn't desirable/feasible either...
import collections.abc
import sys
import os
import math
import itertools
import time
from datetime import datetime

import numpy as np
import larray as la
from larray.util.misc import Product

from larray_editor.utils import Axis, get_sample, scale_to_01range, is_number_value_vectorized
from larray_editor.commands import ArrayValueChange


REGISTERED_ADAPTERS = {}
KB = 2 ** 10
MB = 2 ** 20


# TODO: sadly we cannot use functools.singledispatch because it does not support string types,
#       but the MRO stuff is a lot better than my own code so I could inspire myself with that.
#       Ideally, I could add support for string types in singledispatch and propose the addition
#       to Python
def register_adapter(target_type, adapter_creator):
    """Register an adapter to display a type

    Parameters
    ----------
    target_type : str | type
        Type for which the adapter should be used.
    adapter_creator : callable
        Callable which will return an Adapter instance
    """
    if target_type in REGISTERED_ADAPTERS:
        print(f"Warning: replacing adapter for {target_type}")

    REGISTERED_ADAPTERS[target_type] = adapter_creator


def adapter_for(target_type):
    """Class decorator to register new adapters

    Parameters
    ----------
    target_type : str | type
        Type handled by adapter class.
    """
    def decorate_callable(adapter_creator):
        register_adapter(target_type, adapter_creator)
        return adapter_creator
    return decorate_callable


def get_adapter_creator(data):
    data_type = type(data)
    # first check precise type
    if data_type in REGISTERED_ADAPTERS:
        return REGISTERED_ADAPTERS[data_type]

    adapter_types = list(REGISTERED_ADAPTERS.keys())
    # sort classes with longer MRO first, so that subclasses come before their parent class
    adapter_types.sort(key=lambda cls: 0 if isinstance(cls, str) else -len(cls.mro()))

    # then check subclasses
    for adapter_type in adapter_types:
        adapter_creator = REGISTERED_ADAPTERS[adapter_type]
        if isinstance(adapter_type, str):
            module_name, type_name = adapter_type.split('.', maxsplit=1)
            if module_name in sys.modules:
                obj = sys.modules[module_name]
                while '.' in type_name:
                    attr_name, type_name = type_name.split('.', maxsplit=1)
                    obj = getattr(obj, attr_name)
                # remove string form from adapters mapping
                del REGISTERED_ADAPTERS[adapter_type]

                adapter_type = getattr(obj, type_name)

                # cache precise type adapter if we have more objects of that kind to
                # display later
                REGISTERED_ADAPTERS[adapter_type] = adapter_creator
            else:
                adapter_type = None
        if adapter_type is not None and isinstance(data, adapter_type):
            return adapter_creator
    return None


def get_adapter(data, bg_value):
    if data is None:
        return None
    adapter_creator = get_adapter_creator(data)
    if adapter_creator is None:
        raise TypeError(f"No Adapter implemented for data with type {type(data)}")
    return adapter_creator(data, bg_value)


def nd_shape_to_2d(shape, num_h_axes=1):
    """

    Parameters
    ----------
    shape : tuple
    num_h_axes : int, optional
        Defaults to 1.

    Examples
    --------
    >>> nd_shape_to_2d(())
    (1, 1)
    >>> nd_shape_to_2d((2,))
    (1, 2)
    >>> nd_shape_to_2d((0,))
    (1, 0)
    >>> nd_shape_to_2d((2, 3))
    (2, 3)
    >>> nd_shape_to_2d((2, 0))
    (2, 0)
    >>> nd_shape_to_2d((2, 3, 4))
    (6, 4)
    >>> nd_shape_to_2d((2, 3, 0))
    (6, 0)
    >>> nd_shape_to_2d((2, 0, 4))
    (0, 4)
    >>> nd_shape_to_2d((), num_h_axes=2)
    (1, 1)
    >>> nd_shape_to_2d((2,), num_h_axes=2)
    (1, 2)
    >>> nd_shape_to_2d((2, 3), num_h_axes=2)
    (1, 6)
    >>> nd_shape_to_2d((2, 3, 4), num_h_axes=2)
    (2, 12)
    >>> nd_shape_to_2d((), num_h_axes=0)
    (1, 1)
    >>> nd_shape_to_2d((2,), num_h_axes=0)
    (2, 1)
    >>> nd_shape_to_2d((2, 3), num_h_axes=0)
    (6, 1)
    >>> nd_shape_to_2d((2, 3, 4), num_h_axes=0)
    (24, 1)

    Returns
    -------
    shape: tuple of integers
        2d shape
    """
    shape_v = shape[:-num_h_axes] if num_h_axes else shape
    shape_h = shape[-num_h_axes:] if num_h_axes else ()
    return np.prod(shape_v, dtype=int), np.prod(shape_h, dtype=int)


# CHECK: maybe implement decorator to mark any method as a context menu action. But what we need is not a method
# which does the action, but a method which adds a command part to the current command.

# @context_menu('Transpose')
# def transpose(self):
#     pass

class AbstractAdapter:
    num_h_axes = 1

    # TODO: we should have a way to provide other attributes: format, readonly, font (problematic for colwidth),
    #       align, tooltips, flags?, min_value, max_value (for the delegate), ...
    #       I guess data itself will need to be a dict: {'values': ...}
    def __init__(self, data, bg_value):
        self.data = data
        self.bg_value = bg_value
        # CHECK: filters will probably not make it as-is after quickbar is implemented: they will need to move
        #        to the axes area
        self.current_filter = {}

        # FIXME: this is an ugly/quick&dirty workaround
        # AFAICT, this is only used in ArrayDelegate
        self.dtype = np.dtype(object)
        # self.dtype = None
        # CHECK: unsure this should be part of the API or an implementation detail of ArrayAdapter (or a mixin class?)
        self.vmin = None
        self.vmax = None
        self._number_format = "%s"

    # ================================ #
    # methods which MUST be overridden #
    # ================================ #
    def get_values(self, h_start, v_start, h_stop, v_stop):
        raise NotImplementedError()

    def shape2d(self):
        raise NotImplementedError()

    # =============================== #
    # methods which CAN be overridden #
    # =============================== #
    def get_data(self, h_start, v_start, h_stop, v_stop):
        values = self.get_values(h_start, v_start, h_stop, v_stop)
        return {'data_format': self.get_format(h_start, v_start, h_stop, v_stop),
                'values': values}

    def get_format(self, h_start, v_start, h_stop, v_stop):
        return self._number_format

    def set_format(self, fmt):
        """Change display format"""
        print(f"setting adapter format: {fmt}")
        self._number_format = fmt

    def from_clipboard_data_to_model_data(self, list_data):
        return list_data

    def get_axes_labels_and_data_values(self, row_min, row_max, col_min, col_max):
        axes_names = self.get_axes_area()
        hlabels = self.get_hlabels(col_min, col_max)
        vlabels = self.get_vlabels(row_min, row_max)
        raw_data = self.get_data(col_min, row_min, col_max, row_max)['values']
        return axes_names, vlabels, hlabels, raw_data

    def move_axis(self, data, bg_value, old_index, new_index):
        """Move an axis of the data array and associated bg value.

        Parameters
        ----------
        data : array
            Array to transpose
        bg_value : array or None
            Associated bg_value array.
        old_index: int
            Current index of axis to move.
        new_index: int
            New index of axis after transpose.

        Returns
        -------
        data : array
            Transposed input array
        bg_value: array
            Transposed associated bg_value
        """
        raise NotImplementedError()

    # TODO: split this into get_filter_names and get_filter_values(start, stop)
    #       ... in the end, filters will move to axes names AND possibly h/vlabels and must update the current command
    def get_filters(self):
        """return [(combo_label, combo_values)]"""
        return []

    def update_filter(self, filter_idx, filter_name, indices):
        """Update current filter for a given axis if labels selection from the array widget has changed

        Parameters
        ----------
        axis: Axis
             Axis for which selection has changed.
        indices: list of int
            Indices of selected labels.
        """
        raise NotImplementedError()

    def map_filtered_to_global(self, filtered_shape, filter, local2dkey):
        """
        map local (filtered data) 2D key to global (unfiltered) ND key.

        Parameters
        ----------
        filtered_shape : tuple
            Shape of filtered data.
        filter : dict
            Current filter: {axis_idx: index_or_indices}
        local2dkey: tuple
            Positional index (row, column) of the modified data cell.

        Returns
        -------
        tuple
            ND indices associated with the modified element of the non-filtered array.
        """
        raise NotImplementedError()

    def translate_changes(self, data_model_changes):
        to_global = self.map_filtered_to_global
        # FIXME: filtered_data is specific to LArray. Either make it part of the API, or do not pass it as argument
        #        and get it in the implementation of map_filtered_to_global
        global_changes = [ArrayValueChange(to_global(self.filtered_data.shape, self.current_filter, key),
                                           old_value, new_value)
                          for key, (old_value, new_value) in data_model_changes.items()]
        return global_changes

    def get_sample(self):
        """Return a sample of the internal data"""
        # TODO: use default_buffer sizes instead, or, better yet, a new get_preferred_buffer_size() method
        height, width = self.shape2d()
        return self.get_data(0, 0, min(width, 20), min(height, 20))['values']

    def get_axes_area(self):
        # axes = self.filtered_data.axes
        # test axes.size == 0 is required in case an instance built as Array([]) is passed
        # test len(axes) == 0 is required when a user filters until getting a scalar (because in that case size is 1)
        # TODO: store this in the adapter
        # if axes.size == 0 or len(axes) == 0:
        #     return [[]]
        # else:
        shape = self.shape2d()
        num_h_axes = self.num_h_axes
        row_idx_names = self.get_vnames()
        num_v_axes = len(row_idx_names)
        col_idx_names = self.get_hnames()
        print("names", row_idx_names, col_idx_names)
        if (not len(row_idx_names) and not len(col_idx_names)) or any(d == 0 for d in shape):
            return [[]]
        names = np.full((max(num_h_axes, 1), max(num_v_axes, 1)), '', dtype=object)
        if len(row_idx_names) > 1:
            names[-1, :-1] = row_idx_names[:-1]
        if len(col_idx_names) > 1:
            names[:-1, -1] = col_idx_names[:-1]
        part1 = row_idx_names[-1] if row_idx_names else ''
        part2 = col_idx_names[-1] if col_idx_names else ''
        names[-1, -1] = (part1 + '\\' + part2) if part1 and part2 else part1 + part2
        return names.tolist()

    def get_vlabels(self, start, stop):
        # Note that using some kind of lazy object here is pointless given that
        # we will use most of it (the buffer should not be much larger than the
        # viewport). It would make sense to define one big lazy object as
        # self._vlabels = Product([range(len(data))]) and use return
        # self._vlabels[start:stop] here but I am unsure it is worth it because
        # that would be slower than what we have now.
        return [[i] for i in range(start, stop)]

    def get_hlabels(self, start, stop):
        return [list(range(start, stop))]

    def get_vnames(self):
        return ['']

    def get_hnames(self):
        return ['']

    def get_vname(self):
        return ' '.join(str(name) for name in self.get_vnames())

    def get_hname(self):
        return ' '.join(str(name) for name in self.get_hnames())

    def combine_labels_and_data(self, raw_data, axes_names, vlabels, hlabels):
        """Return list

        Parameters
        ----------
        raw_data : sequence of sequence of built-in scalar types
            Array of selected data. Supports numpy arrays, tuple, list etc.
        axes_names : list of string
            List of axis names
        vlabels : nested list
            Selected vertical labels
        hlabels: nested list
            Selected horizontal labels

        Returns
        -------
        list of list of built-in Python scalars (None, bool, int, float, str)
        """
        # we use itertools.chain so that we can combine any iterables, not just lists
        chain = itertools.chain
        topheaders = [list(chain(axis_row, hlabels_row))
                      for axis_row, hlabels_row in zip(axes_names, hlabels)]
        datarows = [list(chain(row_labels, row_data))
                    for row_labels, row_data in zip(vlabels, raw_data)]
        return topheaders + datarows

    def get_combined_data(self, row_min, row_max, col_min, col_max):
        """Return ...

        Parameters
        ----------

        Returns
        -------
        list of list of built-in Python scalars (None, bool, int, float, str)
        """
        axes_names, vlabels, hlabels, raw_data = self.get_axes_labels_and_data_values(row_min, row_max, col_min, col_max)
        return self.combine_labels_and_data(raw_data, axes_names, vlabels, hlabels)

    def to_string(self, row_min, row_max, col_min, col_max, sep='\t'):
        """Copy selection as tab-separated (clipboard) text

        Returns
        -------
        str
        """
        data = self.get_combined_data(row_min, row_max, col_min, col_max)

        # np.savetxt make things more complicated, especially on py3
        # We do not use repr for everything to avoid having extra quotes for strings.
        # XXX: but is it really a problem? Wouldn't it allow us to copy-paste values with sep (tabs) in them?
        #      I need to test what Excel does for strings
        def vrepr(v):
            if isinstance(v, float):
                return repr(v)
            else:
                return str(v)

        return '\n'.join(sep.join(vrepr(v) for v in line) for line in data)

    def to_excel(self, row_min, row_max, col_min, col_max):
        """Export data to an Excel Sheet
        """
        import xlwings as xw

        data = self.get_combined_data(row_min, row_max, col_min, col_max)
        # convert (row) generators to lists then array
        # TODO: the conversion to array is currently necessary even though xlwings will translate it back to a list
        #       anyway. The problem is that our lists contains numpy types and especially np.str_ crashes xlwings.
        #       unsure how we should fix this properly: in xlwings, or change get_combined_data() to return only
        #       standard Python types.
        array = np.array([list(r) for r in data])
        xw.view(array)

    def plot(self, row_min, row_max, col_min, col_max):
        """Return a matplotlib.Figure object for selected subset.

        Returns
        -------
        A matplotlib.Figure object.
        """
        from matplotlib.figure import Figure

        # we do not use the axes_names part because the position of axes names is up to the adapter
        _, vlabels, hlabels, raw_data = self.get_axes_labels_and_data_values(row_min, row_max, col_min, col_max)
        if not isinstance(raw_data, np.ndarray):
            # TODO: in the presence of a string, raw_data will be entirely converted to strings
            #       which is not what we want. Maybe force object dtype? But that's problematic
            #       for performance. Maybe, we should not rely on numpy arrays. But if
            #       matplotlib converts to it internally anyway we don't gain anything.
            raw_data = np.asarray(raw_data)
            raw_data = raw_data.reshape((raw_data.shape[0], -1))
        assert isinstance(raw_data, np.ndarray), f"got data of type {type(raw_data)}"
        assert raw_data.ndim == 2, f"ndim is {raw_data.ndim}"

        figure = Figure()

        # create an axis
        ax = figure.add_subplot()

        # we have a list of rows but we want a list of columns
        xlabels = list(zip(*hlabels))
        ylabels = vlabels

        xlabel = self.get_hname()
        ylabel = self.get_vname()

        height, width = raw_data.shape
        if width == 1:
            # plot one column
            xlabels, ylabels = ylabels, xlabels
            xlabel, ylabel = ylabel, xlabel
            height, width = width, height
            raw_data = raw_data.T

        # plot each row as a line
        xticklabels = ['\n'.join(str(label) for label in label_col)
                       for label_col in xlabels]
        xdata = np.arange(width)
        for data_row, ylabels_row in zip(raw_data, ylabels):
            ax.plot(xdata, data_row, label=' '.join(str(label) for label in ylabels_row))

        # set x axis
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.set_xlim((0, width - 1))
        # we need to do that because matplotlib is smart enough to
        # not show all ticks but a selection. However, that selection
        # may include ticks outside the range of x axis
        xticks = [t for t in ax.get_xticks().astype(int) if t < len(xticklabels)]
        xticklabels = [xticklabels[t] for t in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # add legend
        all_empty_labels = all(not label for yrow in ylabels for label in yrow)
        if width != 1 and not all_empty_labels:
            kwargs = {'title': ylabel} if ylabel else {}
            ax.legend(**kwargs)

        return figure


class SequenceAdapter(AbstractAdapter):
    def shape2d(self):
        return len(self.data), 1

    def get_axes_area(self):
        return [['index']]

    def get_hlabels(self, start, stop):
        return [['value']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self.data[v_start:v_stop]


class NamedTupleAdapter(AbstractAdapter):
    def shape2d(self):
        return len(self.data), 1

    def get_axes_area(self):
        return [['attribute']]

    def get_hlabels(self, start, stop):
        return [['value']]

    def get_vlabels(self, start, stop):
        return [[k] for k in self.data._fields[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self.data[v_start:v_stop]


@adapter_for(collections.abc.Sequence)
def get_sequence_adapter(data, bg_value):
    namedtuple_attrs = ['_asdict', '_field_defaults', '_fields', '_make', '_replace']
    # Named tuples have no special parent class, so we cannot dispatch using the type
    # of data and need to check the presence of NamedTuple specific attributes instead
    if all(hasattr(data, attr) for attr in namedtuple_attrs):
        return NamedTupleAdapter(data, bg_value)
    else:
        return SequenceAdapter(data, bg_value)


@adapter_for(collections.abc.Mapping)
class MappingAdapter(AbstractAdapter):
    def shape2d(self):
        return len(self.data), 1

    def get_axes_area(self):
        return [['key']]

    def get_hlabels(self, start, stop):
        return [['value']]

    def get_vlabels(self, start, stop):
        # using islice instead of caching list(data.keys()) and list(data.values()) in __init__
        # make things *much* faster to display the first elements of very large dicts at
        # the expense of making the display of the last elements about twice as slow.
        # It seems a desirable tradeoff, especially given the lower memory usage and
        # the absence of stale cache problem. Performance-wise, we could cache keys() and
        # values() here (instead of in __init__) if start or stop is above some threshold
        # but I am unsure it is worth the added complexity.
        return [[k] for k in itertools.islice(self.data.keys(), start, stop)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return list(itertools.islice(self.data.values(), v_start, v_stop))


@adapter_for(collections.abc.Collection)
class CollectionAdapter(AbstractAdapter):
    def shape2d(self):
        return len(self.data), 1

    def get_hlabels(self, start, stop):
        return [['value']]

    def get_vlabels(self, start, stop):
        return [[''] for i in range(start, stop)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return list(itertools.islice(self.data, v_start, v_stop))


# Specific adapter just to change the label
@adapter_for(collections.abc.KeysView)
class KeysViewAdapter(CollectionAdapter):
    def get_hlabels(self, start, stop):
        return [['key']]


@adapter_for(collections.abc.ItemsView)
class ItemsViewAdapter(CollectionAdapter):
    def shape2d(self):
        return len(self.data), 2

    def get_hlabels(self, start, stop):
        return [['key', 'value']]


def get_color_value(sample_array_data, global_vmin, global_vmax, axis=None):
    assert isinstance(sample_array_data, np.ndarray)
    dtype = sample_array_data.dtype
    try:
        color_value = sample_array_data
        # TODO: there are a lot more complex dtypes than this. Is there a way to get them all in one shot?
        if dtype in (np.complex64, np.complex128):
            # for complex numbers, shading will be based on absolute value
            color_value = np.abs(color_value)

        if dtype.type is np.object_:
            color_value = np.where(is_number_value_vectorized(color_value), color_value, np.nan)
            color_value = color_value.astype(np.float64)

        # change inf and -inf to nan (setting them to 0 or to very large numbers is not an option)
        color_value = np.where(np.isfinite(color_value), color_value, np.nan)

        vmin = np.nanmin(color_value, axis=axis)
        if global_vmin is not None:
            # vmin or global_vmin can both be nan (if the whole section data is/was nan)
            global_vmin = np.nanmin([global_vmin, vmin], axis=axis)
        else:
            global_vmin = vmin
        vmax = np.nanmax(color_value, axis=axis)
        if global_vmax is not None:
            # vmax or global_vmax can both be nan (if the whole section data is/was nan)
            global_vmax = np.nanmax([global_vmax, vmax], axis=axis)
        else:
            global_vmax = vmax
        color_value = scale_to_01range(color_value, global_vmin, global_vmax)
    except (ValueError, TypeError):
        global_vmin = None
        global_vmax = None
        color_value = None
    return color_value, global_vmin, global_vmax


@adapter_for(la.Array)
class LArrayArrayAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        # self.num_h_axes = min(data.ndim, 2)
        self.num_h_axes = 1
        data = la.asarray(data)
        bg_value = la.asarray(bg_value) if bg_value is not None else None
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)
        # TODO: should not be needed
        self.dtype = data.dtype

        # TODO: filtered_data should be a dict: {'values': ..., 'bg_value': ..., 'format': ...}
        # XXX: I wonder if that dict should contain 2D numpy arrays or ND la.Arrays.
        #      I think we could need both.
        #      la.Array if we want to extract current selection
        self.filtered_data = self.data
        self.filtered_bg_value = self.bg_value
        self._number_format = "%.3f"

    def from_clipboard_data_to_model_data(self, list_data):
        try:
            # index of first cell which contains '\'
            pos_last = next(i for i, v in enumerate(list_data[0]) if '\\' in v)
        except StopIteration:
            # if there isn't any, assume 1d array
            pos_last = 0

        if pos_last or '\\' in list_data[0][0]:
            # ndim > 1
            # strip horizontal and vertical labels (to keep only the raw data)
            list_data = [line[pos_last + 1:] for line in list_data[1:]]
        elif len(list_data) == 2 and list_data[1][0] == '':
            # ndim == 1, horizontal
            # strip horizontal labels (first line) and empty cell (due to axis name)
            list_data = [list_data[1][1:]]
        else:
            # assume raw data
            pass
        return list_data

    def shape2d(self):
        return nd_shape_to_2d(self.data.shape, num_h_axes=self.num_h_axes)

    def get_filters(self):
        """return [(combo_label, combo_values_sequence)]"""
        axes = self.data.axes
        return [(name, axis.labels) for name, axis in zip(axes.display_names, axes)]

    def _filter_data(self, data, full_indices_filter):
        if data is None:
            return data
        assert isinstance(data, la.Array)
        data = data.i[full_indices_filter]
        return la.asarray(data) if np.isscalar(data) else data

    def update_filter(self, filter_idx, filter_name, indices):
        """Update current filter for a given axis if labels selection from the array widget has changed

        Parameters
        ----------
        filter_idx : int
             Index of filter (axis) for which selection has changed.
        filter_name : str
             Name of filter (axis) for which selection has changed.
        indices: list of int
            Indices of selected labels.
        """
        current_filter = self.current_filter
        axis = self.data.axes[filter_idx]
        if not indices or len(indices) == len(axis):
            if filter_idx in current_filter:
                del current_filter[filter_idx]
        else:
            if len(indices) == 1:
                current_filter[filter_idx] = indices[0]
            else:
                current_filter[filter_idx] = indices

        # current_filter is a {axis_idx: axis_indices} dict
        # full_indices_filter is a tuple
        full_indices_filter = tuple(current_filter[axis_idx] if axis_idx in current_filter else slice(None)
                                    for axis_idx in range(len(self.data.axes)))
        self.filtered_data = self._filter_data(self.data, full_indices_filter)
        self.filtered_bg_value = self._filter_data(self.bg_value, full_indices_filter)

    def get_data(self, h_start, v_start, h_stop, v_stop):
        # data
        # ====
        # get filtered data as Numpy 2D array
        np_data = self.filtered_data.data
        assert isinstance(np_data, np.ndarray)
        shape2d = nd_shape_to_2d(np_data.shape, self.num_h_axes)
        raw_data = np_data.reshape(shape2d)
        section_data = raw_data[v_start:v_stop, h_start:h_stop]

        # bg_value
        # ========
        if self.bg_value is not None:
            # user-provided bg_value
            assert isinstance(self.filtered_bg_value, la.Array)
            bg_value = self.filtered_bg_value.data.reshape(shape2d)
            color_value = bg_value[v_start:v_stop, h_start:h_stop]
        else:
            # "default" bg_value computed on the subset asked by the model
            color_value, self.vmin, self.vmax = get_color_value(section_data, self.vmin, self.vmax)
        return {'editable': True, 'data_format': self._number_format,
                'values': section_data, 'bg_value': color_value}

    def get_vnames(self):
        axes = self.filtered_data.axes
        num_v_axes = max(len(axes) - self.num_h_axes, 0)
        return axes.display_names[:num_v_axes]

    def get_hnames(self):
        axes = self.filtered_data.axes
        num_v_axes = max(len(axes) - self.num_h_axes, 0)
        return axes.display_names[num_v_axes:]

    def get_vlabels(self, start, stop):
        axes = self.filtered_data.axes
        # test data.size == 0 is required in case an instance built as Array([]) is passed
        # test len(axes) == 0 is required when a user filters until getting a scalar (because in that case size is 1)
        # TODO: store this in the adapter
        if axes.size == 0 or len(axes) == 0:
            return [[]]
        elif len(axes) <= self.num_h_axes:
            # all axes are horizontal => a single empty vlabel
            return [['']]
        else:
            # we must not convert the *whole* axes to raw python objects here (e.g. using tolist) because this would be
            # too slow for huge axes
            v_axes = axes[:-self.num_h_axes] if self.num_h_axes else axes
            # CHECK: store self._vlabels in adapter?
            vlabels = Product([axis.labels for axis in v_axes])
            return vlabels[start:stop]

    def get_hlabels(self, start, stop):
        axes = self.filtered_data.axes
        # test data.size == 0 is required in case an instance built as Array([]) is passed
        # test len(axes) == 0 is required when a user filters until to get a scalar
        # TODO: store this in the adapter
        if axes.size == 0 or len(axes) == 0:
            return [[]]
        elif not self.num_h_axes:
            # all axes are vertical => a single empty hlabel
            return [['']]
        else:
            hlabels = Product([axis.labels for axis in axes[-self.num_h_axes:]])
            section_labels = hlabels[start:stop]
            # we have a list of columns but we need a list of rows
            return [[label_col[row_num] for label_col in section_labels]
                    for row_num in range(self.num_h_axes)]

    def get_sample(self):
        """Return a sample of the internal data"""
        np_data = self.filtered_data.data
        # this will yield a data sample of max 200
        return get_sample(np_data, 200)

    def move_axis(self, data, bg_value, old_index, new_index):
        assert isinstance(data, la.Array)
        new_axes = data.axes.copy()
        new_axes.insert(new_index, new_axes.pop(new_axes[old_index]))
        data = data.transpose(new_axes)
        if bg_value is not None:
            assert isinstance(bg_value, la.Array)
            bg_value = bg_value.transpose(new_axes)
        return data, bg_value

    def map_filtered_to_global(self, filtered_shape, filter, local2dkey):
        """
        transform local (filtered) 2D (row_idx, col_idx) key to global (unfiltered) ND key
        (axis0_pos, axis1_pos, ..., axisN_pos). This is positional only (no labels).
        """
        row, col = local2dkey

        localndkey = list(np.unravel_index(row, filtered_shape[:-1])) + [col]

        # add the "scalar" parts of the filter to it (ie the parts of the filter which removed dimensions)
        scalar_filter_keys = [axis_idx for axis_idx, axis_filter in filter.items() if np.isscalar(axis_filter)]
        for axis_idx in sorted(scalar_filter_keys):
            localndkey.insert(axis_idx, filter[axis_idx])

        # translate local to global for filtered dimensions which are still present (non scalar)
        return tuple(
            axis_pos if axis_idx not in filter or np.isscalar(filter[axis_idx]) else filter[axis_idx][axis_pos]
            for axis_idx, axis_pos in enumerate(localndkey)
        )


@adapter_for('array.array')
class ArrayArrayAdapter(AbstractAdapter):
    def shape2d(self):
        return len(self.data), 1

    def get_hlabels(self, start, stop):
        return [['']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self.data[v_start:v_stop].tolist()


def excel_colname(col):
    """col is a *zero* based column number

    >>> excel_colname(0)
    'A'
    >>> excel_colname(25)
    'Z'
    >>> excel_colname(26)
    'AA'
    >>> excel_colname(51)
    'AZ'
    """
    letters = []
    value_a = ord("A")
    while col >= 0:
        letters.append(chr(value_a + col % 26))
        col = (col // 26) - 1
    return "".join(reversed(letters))


@adapter_for('larray.inout.xw_excel.Workbook')
class WorkbookAdapter(SequenceAdapter):
    def __init__(self, data, bg_value):
        SequenceAdapter.__init__(self, data=data.sheet_names(), bg_value=bg_value)

    def get_hlabels(self, start, stop):
        return [['sheet name']]


@adapter_for('larray.inout.xw_excel.Sheet')
class SheetAdapter(AbstractAdapter):
    def shape2d(self):
        return self.data.shape

    def get_hlabels(self, start, stop):
        return [[excel_colname(i) for i in range(start, stop)]]

    def get_vlabels(self, start, stop):
        # +1 because excel rows are 1 based
        return [[i] for i in range(start + 1, stop + 1)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self.data[v_start:v_stop, h_start:h_stop].__array__()


@adapter_for('larray.inout.xw_excel.Range')
class RangeAdapter(AbstractAdapter):
    def shape2d(self):
        return self.data.shape

    def get_hlabels(self, start, stop):
        # - 1 because data.column is 1-based (Excel) while excel_colname is 0-based
        offset = self.data.column - 1
        return [[excel_colname(i) for i in range(offset + start, offset + stop)]]

    def get_vlabels(self, start, stop):
        offset = self.data.row
        return [[i] for i in range(offset + start, offset + stop)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self.data[v_start:v_stop, h_start:h_stop].__array__()


class NumpyHomogeneousArrayAdapter(AbstractAdapter):
    def shape2d(self):
        return nd_shape_to_2d(self.data.shape, num_h_axes=1)

    def get_vnames(self):
        return ['' for axis_len in self.data.shape[:-1]]

    def get_hlabels(self, start, stop):
        if self.data.ndim > 0:
            return [list(range(start, stop))]
        else:
            return [['']]

    def get_vlabels(self, start, stop):
        if self.data.ndim > 0:
            vlabels = Product([range(axis_len) for axis_len in self.data.shape[:-1]])
            return vlabels[start:stop]
        else:
            return [['']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        data2d = self.data.reshape(nd_shape_to_2d(self.data.shape))
        return data2d[v_start:v_stop, h_start:h_stop]


class NumpyStructuredArrayAdapter(AbstractAdapter):
    def shape2d(self):
        shape = self.data.shape + (len(self.data.dtype.names),)
        return nd_shape_to_2d(shape, num_h_axes=1)

    def get_vnames(self):
        return ['' for axis_len in self.data.shape]

    def get_hlabels(self, start, stop):
        return [list(self.data.dtype.names[start:stop])]

    def get_vlabels(self, start, stop):
        vlabels = Product([range(axis_len) for axis_len in self.data.shape])
        return vlabels[start:stop]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        # TODO: this works nicely but isn't any better for users because number of decimals
        #       is not auto-detected and cannot be changed. I think I could implement
        #       auto-detection *relatively* easily but at this point I don't know how
        #       to implement changing it.
        #       One option would be that the ndigits box would set the number of digits for
        #       all *numeric* columns (or even cells?) instead of trying to set it for all columns.
        #       Another option would be that the ndigits box would not be the
        #       number of digits for each column but rather the "bonus" number compared to
        #       the autodetected value.
        #       Yet another option would be to keep track of the number of digits per column
        #       (or cell) and change it only for currently selected cells.
        #       Selecting the entire column would then set it "globally" for the column.
        data1d = self.data.reshape(-1)
        # Each field of a "row" can be accessed via either its name (row['age']) or its position
        # (row[1]) but rows *cannot* be sliced, hence going via tuple(row_data)
        return [tuple(row_data)[h_start:h_stop] for row_data in data1d[v_start:v_stop]]


@adapter_for(np.ndarray)
def get_np_array_adapter(data, bg_value):
    if data.dtype.names is not None:
        return NumpyStructuredArrayAdapter(data, bg_value)
    else:
        return NumpyHomogeneousArrayAdapter(data, bg_value)


@adapter_for('pandas.DataFrame')
class DataFrameAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        import pandas as pd
        globals()['pd'] = pd
        assert isinstance(data, pd.DataFrame)
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)
        self.vmin = None
        self.vmax = None

    def shape2d(self):
        return self.data.shape

    def get_filters(self):
        """return [(combo_label, combo_values_sequence)]"""
        return []
        # df = self.data
        # assert isinstance(df, pd.DataFrame)
        #
        # def get_name_values(index):
        #     if isinstance(index, pd.MultiIndex):
        #         return [(name, labels) for name, labels in zip(index.names, index.levels)]
        #     else:
        #         name = index.name if index.name is not None else ''
        #         return [(name, index.values)]
        #
        # return get_name_values(df.index) + get_name_values(df.columns)
        #

    def filter_data(self, data, filter):
        """
        filter is a {axis_idx: axis_indices} dict
        """
        if data is None or filter is None:
            return data

        assert isinstance(data, pd.DataFrame)
        if isinstance(data.index, pd.MultiIndex) or isinstance(data.columns, pd.MultiIndex):
            print("WARNING: filtering with ndim > 2 not implemented yet")
            return data

        indexer = tuple(filter.get(axis_idx, slice(None)) for axis_idx in range(self.data.ndim))
        res = data.iloc[indexer]
        if isinstance(res, pd.Series):
            res = res.to_frame()
        return res

    # FIXME: this is currently required but should not be (=> need to fix super.get_axes_area to make it work)
    def get_axes_area(self):
        idx_names = self.get_vnames()
        col_names = self.get_hnames()

        names = np.full((len(col_names), len(idx_names)), '', dtype=object)
        names[-1, :-1] = idx_names[:-1]
        names[:-1, -1] = col_names[:-1]
        names[-1, -1] = idx_names[-1] + '\\' + col_names[-1]
        return names.tolist()

    # TODO: maybe support None values natively so that this could just be "return self.data.columns.names"
    def get_hnames(self):
        return [name if name is not None else ''
                for name in self.data.columns.names]

    def get_vnames(self):
        return [name if name is not None else ''
                for name in self.data.index.names]

    def get_vlabels(self, start, stop):
        index = self.data.index[start:stop]
        if isinstance(index, pd.MultiIndex):
            return index.values
        else:
            return index.values[:, np.newaxis]

    def get_hlabels(self, start, stop):
        index = self.data.columns[start:stop]
        if isinstance(index, pd.MultiIndex):
            return [index.get_level_values(i).values for i in range(index.nlevels)]
        else:
            return [index.values]

    def get_data(self, h_start, v_start, h_stop, v_stop):
        section_data = self.data.iloc[v_start:v_stop, h_start:h_stop].values
        color_value, self.vmin, self.vmax = get_color_value(section_data, self.vmin, self.vmax, axis=0)
        return {'data_format': self._number_format, 'values': section_data, 'bg_value': color_value}


@adapter_for('pandas.Series')
class SeriesAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        import pandas as pd
        globals()['pd'] = pd

        assert isinstance(data, pd.Series)
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)

    def shape2d(self):
        return len(self.data), 1

    def get_axes_area(self):
        return [self.get_vnames()]

    def get_vnames(self):
        return [name if name is not None else ''
                for name in self.data.index.names]

    def get_vlabels(self, start, stop):
        index = self.data.index[start:stop]
        if isinstance(index, pd.MultiIndex):
            return index.values
        else:
            return index.values[:, np.newaxis]

    def get_hlabels(self, start, stop):
        return [['']]

    def get_data(self, h_start, v_start, h_stop, v_stop):
        assert h_start == 0
        # h_stop can be > 0 but should be ignored

        # TODO: use self.data.to_dict('list') instead of .values?
        # Note that using .values gives us an object dtype for mixed type series, which is what we want.
        # It is faster than .to_dict('list') for largish structures, but for smallish (e.g. 50, 4), it is slower,
        # so I need to benchmark using full buffer size dataframes
        section_data = self.data.iloc[v_start:v_stop].values.reshape(-1, 1)
        color_value, self.vmin, self.vmax = get_color_value(section_data, self.vmin, self.vmax)
        return {'data_format': self._number_format, 'values': section_data, 'bg_value': color_value}


@adapter_for('pyarrow.Array')
class PyArrowArrayAdapter(AbstractAdapter):
    def shape2d(self):
        return len(self.data), 1

    def get_hlabels(self, start, stop):
        return [['value']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        # TODO: use to_numpy instead??
        return self.data[v_start:v_stop].to_pylist()


@adapter_for('pyarrow.Table')
class PyArrowTableAdapter(AbstractAdapter):
    def shape2d(self):
        return self.data.shape

    def get_hlabels(self, start, stop):
        return [self.data.column_names[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        # FIXME: take h_start and h_stop into account
        return list(zip(*[col.to_pylist()
                          for col in self.data[v_start:v_stop].itercolumns()]))


@adapter_for('polars.DataFrame')
class PolarsDataFrameAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)
        self.vmin = None
        self.vmax = None

    @property
    def shape2d(self):
        return self.data.shape

    def get_hlabels(self, start, stop):
        return [self.data.columns[start:stop]]

    def get_data(self, h_start, v_start, h_stop, v_stop):
        sub_df = self.data[v_start:v_stop, h_start:h_stop]

        buf = np.empty((len(sub_df), len(sub_df.columns)), dtype=object)
        for i, col in enumerate(sub_df.columns):
            buf[:, i] = sub_df[col]

        color_value, self.vmin, self.vmax = get_color_value(buf, self.vmin, self.vmax, axis=0)
        return {'values': buf, 'bg_value': color_value}


# TODO: reuse NumpyStructuredArrayAdapter
# TODO: this does not work super nicely for pandas tables (which store all compatible numeric columns in a single
#       "array" column values_block0
@adapter_for('tables.Table')
class PytablesTableAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)

    def shape2d(self):
        return len(self.data), len(self.data.dtype.names)

    def get_hlabels(self, start, stop):
        return [list(self.data.dtype.names[start:stop])]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        # TODO: when we scroll horizontally, we fetch the data over and over while we could only fetch it once
        #       given that pytables fetches entire rows anyway. Several solutions:
        #       * cache "current" rows in the adapter
        #       * have a way for the arraymodel to ask the adapter for the minimum buffer size
        #       * allow the adapter to return more data than what the model asked for and have the model actually
        #         use/take that extra data into account. This would require the adapter to return
        #         real_h_start, real_v_start (stop values can be deduced) in addition to actual values
        names = self.get_hlabels(h_start, h_stop)[0]
        return self.data[v_start:v_stop][names]


# TODO: options to display as hex or decimal
# >>> s = f.read(20)
# >>> s
# b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc2\xea\x81\xb3\x14\x11\xcf\xbd
@adapter_for('io.BufferedReader')
class BinaryFileAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)
        self._nbytes = os.path.getsize(data.name)
        self._width = 16

    def shape2d(self):
        return math.ceil(self._nbytes / self._width), self._width

    def get_vlabels(self, start, stop):
        start, stop, step = slice(start, stop).indices(self.shape2d()[0])
        return [[i * self._width] for i in range(start, stop)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        f = self.data
        width = self._width

        backup_pos = f.tell()

        # read data (ignoring horizontal bounds at this point)
        start_pos = v_start * width
        stop_pos = v_stop * width
        f.seek(start_pos)
        s = f.read(stop_pos - start_pos)

        # restore file position
        f.seek(backup_pos)

        # load the string as an array of unsigned bytes
        buffer1d = np.frombuffer(s, dtype='u1')

        # enlarge the array so that it is divisible by width (so that we can reshape it)
        buffer_size = len(buffer1d)
        size_remainder = buffer_size % width
        if size_remainder != 0:
            filler_size = width - size_remainder
            rounded_size = buffer_size + filler_size
            try:
                # first try inplace resize
                buffer1d.resize(rounded_size, refcheck=False)
            except:
                buffer1d = np.append(buffer1d, np.zeros(filler_size, dtype='u1'))

        # change undisplayable characters to '.'
        buffer1d = np.where((buffer1d < 32) | (buffer1d >= 128), ord('.'), buffer1d).view('S1')

        # reshape to 2d
        buffer2d = buffer1d.reshape((-1, width))

        # take what we were asked for
        return buffer2d[:, h_start:h_stop]


def index_line_ends(s, index=None, offset=0, c='\n'):
    r"""returns a list of line end positions

    It does NOT add an implicit line end at the end of the string.

    >>> index_line_ends("0\n234\n6\n8")
    [1, 5, 7]
    >>> chunks = ["0\n234\n6", "", "\n", "8"]
    >>> pos = 0
    >>> idx = []
    >>> for chunk in chunks:
    ...     _ = index_line_ends(chunk, idx, pos)
    ...     pos += len(chunk)
    >>> idx
    [1, 5, 7]
    """
    if index is None:
        index = []
    if not len(s):
        return index
    line_start = 0
    find = s.find
    append = index.append
    while True:
        line_end = find(c, line_start)
        if line_end == -1:
            break
        append(line_end + offset)
        line_start = line_end + 1
    return index


def chunks_to_lines(chunks, num_lines_required=None):
    r"""
    Parameters
    ----------
    chunks : list
        List of chunks. str and bytes are both supported but should not be mixed (all chunks must
        have the same type than the first chunk).

    Examples
    --------
    >>> chunks = ['a\nb\nc ', 'c\n', 'd ', 'd', '\n', 'e']
    >>> chunks_to_lines(chunks)
    ['a', 'b', 'c c', 'd d', 'e']
    >>> # it should have the same result than join then splitlines (just do it more efficiently)
    ... ''.join(chunks).splitlines()
    ['a', 'b', 'c c', 'd d', 'e']
    """
    if not chunks:
        return []
    sep = b'' if isinstance(chunks[0], bytes) else ''
    lines = sep.join(chunks).splitlines()
    return lines[:num_lines_required]


PATH_ADAPTERS = {}


def register_path_adapter(suffix, adapter_creator, required_module=None):
    """Register an adapter to display a file type (extension)

    Parameters
    ----------
    suffix : str
        File type for which the adapter should be used.
    adapter_creator : callable
        Callable which will return an Adapter instance
    required_module : str
        Name of module required to handle this file type.
    """
    if suffix in PATH_ADAPTERS:
        print(f"Warning: replacing adapter for {suffix}")
    PATH_ADAPTERS[suffix] = (adapter_creator, required_module)


def path_adapter_for(suffix, required_module=None):
    """Class/function decorator to register new file-type adapters

    Parameters
    ----------
    suffix : str
        File type associated with adapter class.
    required_module : str
        Name of module required to handle this file type.
    """
    def decorate_callable(adapter_creator):
        register_path_adapter(suffix, adapter_creator, required_module)
        return adapter_creator
    return decorate_callable


@path_adapter_for('.txt')
@path_adapter_for('.py')
@path_adapter_for('.yml')
@adapter_for('io.TextIOWrapper')
class TextFileAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)
        self._nbytes = os.path.getsize(self._path)
        self._lines_end_index = []
        self._fully_indexed = False

        # sniff a small chunk so that we can compute an approximate number of lines
        with self._binary_file as f:
            self._index_up_to(f, 1, chunk_size=64 * KB, max_time=0.05)

    @property
    def _path(self):
        import io
        return self.data.name if isinstance(self.data, io.TextIOWrapper) else self.data

    @property
    def _binary_file(self):
        return open(self._path, 'rb')

    @property
    def _avg_bytes_per_line(self):
        return self._lines_end_index[-1] / len(self._lines_end_index)

    @property
    def _num_lines(self):
        """returns estimated number of lines"""
        if self._fully_indexed:
            return len(self._lines_end_index)
        else:
            return math.ceil(self._nbytes / self._avg_bytes_per_line)

    def shape2d(self):
        return self._num_lines, 1

    def _index_up_to(self, f, approx_v_stop, chunk_size=4 * MB, max_time=0.5):
        # If the size of the index ever becomes a problem, we could store only
        # one line on X but we are not there yet.
        # We also need to limit line length (to something like 256Kb?). Beyond that it is
        # probably not a line-based file.
        if len(self._lines_end_index):
            lines_to_index = max(approx_v_stop - len(self._lines_end_index), 0)
            data_to_index = lines_to_index * self._avg_bytes_per_line
            must_index = 0 < data_to_index < 512 * MB
        else:
            # we have not indexed anything yet
            must_index = True

        if must_index:
            print(f"trying to index up to {approx_v_stop}...")
            start_time = time.perf_counter()
            chunk_start = self._lines_end_index[-1] if self._lines_end_index else 0
            f.seek(chunk_start)
            # TODO: check for off by one error with v_stop
            while (time.perf_counter() - start_time < max_time) and (len(self._lines_end_index) < approx_v_stop) and \
                    not self._fully_indexed:

                # TODO: if we are beyond v_start, we should store the chunks to avoid reading them twice from disk
                #       (once for indexing then again for getting the data)
                chunk = f.read(chunk_size)

                line_end_char = b'\n'
                index_line_ends(chunk, self._lines_end_index, offset=chunk_start, c=line_end_char)
                length_read = len(chunk)
                if length_read < chunk_size:
                    self._fully_indexed = True
                    # add implicit line end at the end of the file if there isn't an explicit one
                    file_length = chunk_start + length_read
                    file_last_char_pos = file_length - len(line_end_char)
                    if self._lines_end_index[-1] != file_last_char_pos:
                        self._lines_end_index.append(file_length)
                chunk_start += length_read

            # TODO: check for off by one error with v_stop
            if len(self._lines_end_index) < approx_v_stop and not self._fully_indexed:
                print(f" > timed out! indexed up to {len(self._lines_end_index)} but needed {approx_v_stop}")
            else:
                print(" > got it!")

    def get_vlabels(self, start, stop):
        # we need to trigger indexing too (because get_vlabels happens before get_data) so that lines_indexed is correct
        # FIXME: get_data should not trigger indexing too if start/stop are the same
        with self._binary_file as f:
            self._index_up_to(f, stop)

        start, stop, step = slice(start, stop).indices(self._num_lines)
        lines_indexed = len(self._lines_end_index)
        return [[str(i) if i < lines_indexed else '~' + str(i)] for i in range(start, stop)]

    def _get_lines(self, start, stop):
        """stop is exclusive"""
        print(f"_get_lines {start}:{stop}")
        assert start >= 0 and stop >= 0
        with self._binary_file as f:
            self._index_up_to(f, stop)
            num_indexed_lines = len(self._lines_end_index)
            if self._fully_indexed and stop > num_indexed_lines:
                stop = num_indexed_lines

            # if we are entirely in indexed lines, we can use exact pos
            if stop <= num_indexed_lines:
                # position of first line is one byte after the end of the line preceding it (if any)
                start_pos = self._lines_end_index[start - 1] + 1 if start >= 1 else 0
                # v_stop line should be excluded (=> -1)
                stop_pos = self._lines_end_index[stop - 1]
                f.seek(start_pos)
                chunk = f.read(stop_pos - start_pos)
                lines = chunk.split(b'\n')
                assert len(lines) == stop - start
                return lines
            else:
                pos_last_end = self._lines_end_index[-1]
                if start - 1 < num_indexed_lines:
                    approx_start = False
                    start_pos = self._lines_end_index[start - 1] + 1 if start >= 1 else 0
                else:
                    approx_start = True
                    # use approximate pos for start
                    start_pos = pos_last_end + 1 + int((start - num_indexed_lines) * self._avg_bytes_per_line)
                    # read one more line before expected start_pos to have more chance of getting the line entirely
                    start_pos = max(start_pos - int(self._avg_bytes_per_line), 0)

                num_lines = 0
                num_lines_required = stop - start

                f.seek(start_pos)
                # use approximate pos for stop
                chunks = []
                CHUNK_SIZE = 1 * MB
                stop_pos = pos_last_end + math.ceil((stop - num_indexed_lines) * self._avg_bytes_per_line)
                max_stop_pos = min(stop_pos + 4 * MB, self._nbytes)
                # first chunk size is what we *think* is necessary to get num_lines_required
                chunk_size = stop_pos - start_pos
                # but then, if the number of lines we actually got (num_lines) is not enough we will ask for more
                while num_lines < num_lines_required and stop_pos < max_stop_pos:
                    chunk = f.read(chunk_size)
                    chunks.append(chunk)
                    num_lines += chunk.count(b'\n')
                    stop_pos += len(chunk)
                    chunk_size = CHUNK_SIZE

                if approx_start:
                    # +1 and [1:] to remove first line so that we are sure the first line is complete
                    lines = chunks_to_lines(chunks, num_lines_required + 1)[1:]
                else:
                    lines = chunks_to_lines(chunks, num_lines_required)
                return lines

    def get_values(self, h_start, v_start, h_stop, v_stop):
        """*_stop are exclusive"""
        return self._get_lines(v_start, v_stop)


@path_adapter_for('.csv', 'csv')
class CsvPathAdapter(TextFileAdapter):
    def __init__(self, data, bg_value):
        # we know the module is loaded but it is not in the current namespace
        csv = sys.modules['csv']
        data = str(data)
        TextFileAdapter.__init__(self, data=data, bg_value=bg_value)
        first_line = self._get_lines(0, 1)
        # TODO: gracefully handle empty file
        assert len(first_line) == 1
        reader = csv.reader([first_line[0].decode('utf8')])
        self._colnames = next(reader)

    # note that for large files, this is approximate
    def shape2d(self):
        # - 1 for header row
        return self._num_lines - 1, len(self._colnames)

    def get_hlabels(self, start, stop):
        return [self._colnames[start:stop]]

    def get_vlabels(self, start, stop):
        # + 1 for header row
        return super().get_vlabels(start + 1, stop + 1)

    def get_values(self, h_start, v_start, h_stop, v_stop):
        """*_stop are exclusive"""
        print("get_values", h_start, v_start, h_stop, v_stop)

        # + 1 because the header row is not part of the data but _get_lines works
        # on the actual file lines
        lines = self._get_lines(v_start + 1, v_stop + 1)
        if not lines:
            return []
        # we know the module is loaded but it is not in the current namespace
        csv = sys.modules['csv']
        # Note that csv reader actually needs a line-based input
        reader = csv.reader([line.decode('utf8') for line in lines])
        return [line[h_start:h_stop] for line in reader]


@path_adapter_for('.sas7bdat', 'pyreadstat')
class Sas7BdatPathAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        # we know the module is loaded but it is not in the current namespace
        pyreadstat = sys.modules['pyreadstat']
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)
        df, meta = pyreadstat.read_sas7bdat(str(data), metadataonly=True)
        self._colnames = meta.column_names
        self._numrows = meta.number_rows

    def shape2d(self):
        return self._numrows, len(self._colnames)

    def get_hlabels(self, start, stop):
        return [self._colnames[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        # we know the module is loaded but it is not in the current namespace
        pyreadstat = sys.modules['pyreadstat']
        df, meta = pyreadstat.read_sas7bdat(str(self.data), row_offset=v_start, row_limit=v_stop - v_start)
        return df.iloc[:, h_start:h_stop].values


class DirectoryPathAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)
        path_objs = list(data.iterdir())
        # sort by type then name
        path_objs.sort(key=lambda p: (not p.is_dir(), p.name))
        stat_objs = [os.stat(p) for p in path_objs]
        self._list = [(p.name,
                       datetime.fromtimestamp(s.st_mtime).strftime('%d/%m/%Y %H:%M'),
                       '<directory>' if p.is_dir() else s.st_size)
                      for p, s in zip(path_objs, stat_objs)]
        self._colnames = ['Name', 'Time Modified', 'Size']

    def shape2d(self):
        return len(self._list), len(self._colnames)

    def get_hlabels(self, start, stop):
        return [self._colnames[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self._list[v_start:v_stop]


@adapter_for('pathlib.Path')
def get_path_adapter(data, bg_value):
    print("get_path_adapter", data, bg_value)
    print("suffix", data.suffix)
    if data.suffix in PATH_ADAPTERS:
        cls, required_module = PATH_ADAPTERS[data.suffix]
        if required_module is not None:
            if required_module not in sys.modules:
                import importlib
                try:
                    importlib.import_module(required_module)
                except ImportError:
                    return None
        return cls(data, bg_value)
    elif data.is_dir():
        return DirectoryPathAdapter(data, bg_value)
    else:
        return None


@adapter_for('pstats.Stats')
class ProfilingStatsAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)
        self._keys = list(data.stats.keys())
        self._colnames = ['filepath', 'line num', 'func. name', 'ncalls (non rec)', 'ncalls (total)', 'tottime',
                          'cumtime']

    def shape2d(self):
        return len(self._keys), len(self._colnames)

    def get_hlabels(self, start, stop):
        return [self._colnames[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        """*_stop are exclusive"""

        func_calls = self._keys[v_start:v_stop]
        stats = self.data.stats
        call_details = [stats[k] for k in func_calls]
        return [(filepath, line_num, func_name, ncalls_primitive, ncalls_tot, tottime, cumtime)[h_start:h_stop]
                for ((filepath, line_num, func_name), (ncalls_primitive, ncalls_tot, tottime, cumtime, callers))
                in zip(func_calls, call_details)]


SQLITE_LIST_TABLES_QUERY = "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'"


class SQLiteTable:
    def __init__(self, con, name):
        self.con = con
        self.name = name

    def __repr__(self):
        return f"<SQLiteTable '{self.name}'>"


class SQLiteExplorer:
    def __init__(self, con):
        self.con = con

    def __dir__(self):
        cur = self.con.cursor()
        cur.execute(SQLITE_LIST_TABLES_QUERY)
        rows = cur.fetchall()
        cur.close()
        return [row[0] for row in rows]

    def __getattr__(self, item):
        if item not in self.__dir__():
            raise AttributeError(f"Database does not contain any '{item}' table")
        return SQLiteTable(self.con, item)

    def __repr__(self):
        return f"<SQLiteExplorer>"


@adapter_for(SQLiteExplorer)
def get_sqlite_explorer_adapter(data: SQLiteExplorer, bg_value):
    return SQLiteConnectionAdapter(data.con, bg_value)


@adapter_for(SQLiteTable)
class SQLiteTableAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)
        cur = self.data.con.cursor()
        cur.execute(f"SELECT count(*) FROM {self.data.name}")
        self._numrows = cur.fetchone()[0]
        cur.execute(f"SELECT * FROM {self.data.name} LIMIT 1")
        self._columns = [col_descr[0] for col_descr in cur.description]
        cur.close()

    def shape2d(self):
        return self._numrows, len(self._columns)

    def get_hlabels(self, start, stop):
        return [self._columns[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        cur = self.data.con.cursor()
        cols = self._columns[h_start:h_stop]
        cur.execute(f"SELECT {', '.join(cols)} FROM {self.data.name} LIMIT {v_stop - v_start} OFFSET {v_start}")
        rows = cur.fetchall()
        cur.close()
        return rows


@adapter_for('sqlite3.Connection')
class SQLiteConnectionAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)
        cur = data.cursor()
        cur.execute(SQLITE_LIST_TABLES_QUERY)
        self._tables = cur.fetchall()
        cur.close()

    def shape2d(self):
        return len(self._tables), 1

    def get_hlabels(self, start, stop):
        return [['Name']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self._tables[v_start:v_stop]


@adapter_for('zipfile.ZipFile')
class ZipFileAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)
        infolist = data.infolist()
        self._list = [(info.filename,
                       datetime(*info.date_time).strftime('%d/%m/%Y %H:%M'),
                       '<directory>' if info.is_dir() else info.file_size)
                      for info in infolist]
        self._colnames = ['Name', 'Time Modified', 'Size']

    def shape2d(self):
        return len(self._list), len(self._colnames)

    def get_hlabels(self, start, stop):
        return [self._colnames[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self._list[v_start:v_stop]


@adapter_for('zipfile.Path')
class ZipPathAdapter(AbstractAdapter):
    def __init__(self, data, bg_value):
        AbstractAdapter.__init__(self, data=data, bg_value=bg_value)
        zpath_objs = list(data.iterdir())
        zpath_objs.sort(key=lambda p: (not p.is_dir(), p.name))
        self._list = [(p.name, '<DIR>' if p.is_dir() else '')
                      for p in zpath_objs]
        self._colnames = ['Name', 'Type']

    def shape2d(self):
        return len(self._list), len(self._colnames)

    def get_hlabels(self, start, stop):
        return [self._colnames[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self._list[v_start:v_stop]
