import collections.abc
import sys
import itertools

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
