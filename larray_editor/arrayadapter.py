# FIXME: * drag and drop axes uses set_data while changing filters do not

# TODO:
# redesign (again) the adapter <> arraymodel boundary:
#   - the adapter may return buffers of any size (the most efficient size
#     which includes the requested area). It must include the requested area if
#     it exists. The buffer must be reasonably small (must fit in RAM
#     comfortably). In that case, the adapter must also return actual hstart
#     and vstart.
#     >>> on second thoughts, I am unsure this is a good idea. It might be
#         better to store the entire buffer on the adapter and have a
#         BufferedAdapter base class (or maybe do this in AbstractAdapter
#         directly -- but doing this for in-memory containers is wasteful).
#   - the buffers MUST be 2D
#   - what about type? numpy or any sequence?
# * we should always have 2 buffers worth in memory
#   - asking for a new buffer/chunk should be done in a Thread
#   - when there are less than X lines unseen, ask for more. X should depend on
#     size of buffer and time to fetch a new buffer
# TODO (long term): add support for streaming data source. In that case,
#      the behavior should be mostly what we had before (when we scroll, it
#      requests more data and the total length of the scrollbar is updated)
#
# TODO (even longer term): add support for streaming data source with a limit
#      (ie keep last N entries)
#
# TODO: add support for "progressive" data sources, e.g. pandas SAS reader
#       (from pandas.io.sas.sas7bdat import SAS7BDATReader), which can read by
#       chunks but cannot read a particular offset. It would be crazy to
#       re-read the whole thing up to the requested data each time, but caching
#       the whole file in memory probably isn't desirable/feasible either, so
#       I guess the best we can do is to cache as many chunks as we can without
#       filling up the memory (the first chunk + the last few we just read
#       are probably the most likely to be re-visited) and read from the file
#       if the user requests some data outside of those chunks
import collections.abc
import logging
import sys
import os
import math
import itertools
import time
# import types
from datetime import datetime
from typing import Optional
from pathlib import Path

import numpy as np
import larray as la
from larray.util.misc import Product

from larray_editor.utils import (get_sample, scale_to_01range,
                                 is_number_value_vectorized, logger)
from larray_editor.commands import CellValueChange


def indirect_sort(seq, ascending):
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=not ascending)


REGISTERED_ADAPTERS = {}
REGISTERED_ADAPTERS_USING_STRINGS = {}
REGISTERED_ADAPTER_TYPES = None
KB = 2 ** 10
MB = 2 ** 20


def register_adapter_using_string(target_type: str, adapter_creator):
    """Register an adapter to display a type

    Parameters
    ----------
    target_type : str
        Type for which the adapter should be used, given as a string.
    adapter_creator : callable
        Callable which will return an Adapter instance
    """
    assert '.' in target_type
    top_module_name, type_name = target_type.split('.', maxsplit=1)
    module_adapters = REGISTERED_ADAPTERS_USING_STRINGS.setdefault(top_module_name, {})
    if type_name in module_adapters:
        logger.warning(f"Replacing adapter for {target_type}")
    module_adapters[type_name] = adapter_creator
    # container = REGISTERED_ADAPTERS_USING_STRINGS
    # parts = target_type.split('.')
    # for i, p in enumerate(parts):
    #     if i == len(parts) - 1:
    #         if p in container:
    #             print(f"Warning: replacing adapter for {target_type}")
    #         container[p] = adapter_creator
    #     else:
    #         container = container.setdefault(p, {})

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
    if isinstance(target_type, str):
        register_adapter_using_string(target_type, adapter_creator)
        return

    if target_type in REGISTERED_ADAPTERS:
        logger.warning(f"Warning: replacing adapter for {target_type}")

    REGISTERED_ADAPTERS[target_type] = adapter_creator

    # normally, the list is created only once when a first adapter is
    # asked for, but if an adapter is registered after that point we need
    # to update the list
    if REGISTERED_ADAPTER_TYPES is not None:
        update_registered_adapter_types()


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


PATH_SUFFIX_ADAPTERS = {}


def register_path_adapter(suffix, adapter_creator, required_module=None):
    """Register an adapter to display a file type (extension)

    Parameters
    ----------
    suffix : str
        File type for which the adapter should be used.
    adapter_creator : callable
        Callable which will return an Adapter instance.
    required_module : str
        Name of module required to handle this file type.
    """
    if suffix in PATH_SUFFIX_ADAPTERS:
        logger.warning(f"Replacing path adapter for {suffix}")
    PATH_SUFFIX_ADAPTERS[suffix] = (adapter_creator, required_module)


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


def get_adapter_creator_for_type(data_type):
    # first check precise type
    if data_type in REGISTERED_ADAPTERS:
        return REGISTERED_ADAPTERS[data_type]

    data_type_full_module_name = data_type.__module__
    if '.' in data_type_full_module_name:
        data_type_top_module_name, _ = data_type_full_module_name.split('.', maxsplit=1)
    else:
        data_type_top_module_name = data_type_full_module_name

    # handle string types
    if data_type_top_module_name in REGISTERED_ADAPTERS_USING_STRINGS:
        assert data_type_top_module_name in sys.modules
        module = sys.modules[data_type_top_module_name]
        module_adapters = REGISTERED_ADAPTERS_USING_STRINGS[data_type_top_module_name]
        # register all adapters for that module using concrete types (instead
        # of string types)
        for str_adapter_type, adapter in list(module_adapters.items()):
            # submodule
            type_name = str_adapter_type
            while '.' in type_name and module is not None:
                submodule_name, type_name = type_name.split('.', maxsplit=1)
                module = getattr(module, submodule_name, None)
                # submodule not found (probably not loaded yet)
                if module is None:
                    continue

            adapter_type = getattr(module, type_name, None)
            if adapter_type is None:
                continue

            # cache real adapter type if we have (more) objects of that kind to
            # display later
            REGISTERED_ADAPTERS[adapter_type] = adapter

            update_registered_adapter_types()

            # remove string form from adapters mapping
            del module_adapters[str_adapter_type]
            if not module_adapters:
                del REGISTERED_ADAPTERS_USING_STRINGS[data_type_top_module_name]

    # then check subclasses
    if REGISTERED_ADAPTER_TYPES is None:
        update_registered_adapter_types()

    for adapter_type in REGISTERED_ADAPTER_TYPES:
        if issubclass(data_type, adapter_type):
            return REGISTERED_ADAPTERS[adapter_type]
    return None


def get_adapter_creator(data):
    obj_type = type(data)
    creator = get_adapter_creator_for_type(obj_type)
    # 3 options:
    # - the type is not handled
    if creator is None:
        return None
    # - all instances of the type are handled by the same adapter
    elif isinstance(creator, type) and issubclass(creator, AbstractAdapter):
        return creator
    # - different adapters handle that type and/or not all instance are handled
    else:
        return creator(data)


def update_registered_adapter_types():
    global REGISTERED_ADAPTER_TYPES

    REGISTERED_ADAPTER_TYPES = list(REGISTERED_ADAPTERS.keys())
    # sort classes with longer MRO first, so that subclasses come before
    # their parent class
    def class_mro_length(cls):
        return len(cls.mro())

    REGISTERED_ADAPTER_TYPES.sort(key=class_mro_length, reverse=True)


def get_adapter(data, attributes=None):
    if data is None:
        return None
    adapter_creator = get_adapter_creator(data)
    if adapter_creator is None:
        raise TypeError(f"No Adapter implemented for data with type {type(data)}")
    resource_handler = adapter_creator.open(data)
    return adapter_creator(resource_handler, attributes)


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
    # TODO: we should have a way to provide other attributes: format, readonly, font (problematic for colwidth),
    #       align, tooltips, flags?, min_value, max_value (for the delegate), ...
    #       I guess data itself will need to be a dict: {'values': ...}
    def __init__(self, data, attributes=None):
        self.data = data
        self.attributes = attributes
        # CHECK: filters will probably not make it as-is after quickbar is implemented: they will need to move
        #        to the axes area
        #        AND possibly h/vlabels and
        #        must update the current command
        self.current_filter = {}
        self._current_sort = []

        # FIXME: this is an ugly/quick&dirty workaround
        # AFAICT, this is only used in ArrayDelegate
        self.dtype = np.dtype(object)
        # self.dtype = None
        self.vmin = None
        self.vmax = None
        self._number_format = "%s"
        self.sort_key = None  # (kind='axis'|'column'|'row', idx_of_kind, direction (1, -1))

    # ================================ #
    # methods which MUST be overridden #
    # ================================ #
    # def get_values(self, h_start, v_start, h_stop, v_stop):
    #     raise NotImplementedError()

    # TODO: split this into:
    #         - extract_chunk_from_data (result is in native/cheapest
    #           format to produce)
    #       and
    #         - native_chunk_to_2D_sequence
    #      the goal is to cache chunks
    def get_chunk_from_data(self, data, h_start, v_start, h_stop, v_stop):
        """
        Extract a subset of a data object of the type the adapter handles.
        Must return a 2D sequence, preferably a numpy array.
        """
        raise NotImplementedError()

    def shape2d(self):
        raise NotImplementedError()

    # =============================== #
    # methods which CAN be overridden #
    # =============================== #

    def cell_activated(self, row_idx, column_idx):
        """
        If this method returns a (not None) value, it will be used as the new
        value for the array_editor_widget. Later this should add an operand on
        the quickbar but we are not there yet.
        """
        return None

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self.get_chunk_from_data(self.data, h_start, v_start, h_stop, v_stop)

    @classmethod
    def open(cls, data):
        """Open the ressources used by the adapter

        The result of this method will be stored in the .data
        attribute and passed as argument to the adapter class"""
        return data

    def close(self):
        """Close the ressources used by the adapter"""
        pass

    # TODO: factorize with LArrayArrayAdapter (so that we get the attributes
    #       handling of LArrayArrayAdapter for all types and the larray adapter
    #       can benefit from the generic code here
    def get_data_values_and_attributes(self, h_start, v_start, h_stop, v_stop):
        """h_stop and v_stop should *not* be included"""
        # TODO: implement region caching
        logger.debug(
            f"{self.__class__.__name__}.get_data_values_and_attributes("
            f"{h_start=}, {v_start=}, {h_stop=}, {v_stop=})"
        )
        chunk_values = self.get_values(h_start, v_start, h_stop, v_stop)
        if isinstance(chunk_values, np.ndarray):
            assert chunk_values.ndim == 2
            logger.debug(f"    {chunk_values.shape=}")
        elif isinstance(chunk_values, list) and len(chunk_values) == 0:
            chunk_values = [[]]

        # Without specifying dtype=object, asarray converts sequences
        # containing both strings and numbers to all strings which then
        # fail in get_color_value, but we do not want to convert
        # existing numpy arrays to object dtype. This is a bit silly and
        # inefficient for numeric-only sequences, but I do not see
        # a better way.
        if not isinstance(chunk_values, np.ndarray):
            chunk_values = np.asarray(chunk_values, dtype=object)
        finite_values = get_finite_numeric_values(chunk_values)
        vmin, vmax = self.update_finite_min_max_values(finite_values,
                                                       h_start, v_start,
                                                       h_stop, v_stop)
        color_value = scale_to_01range(finite_values, vmin, vmax)
        chunk_format = self.get_format(chunk_values, h_start, v_start, h_stop, v_stop)
        return {'data_format': chunk_format,
                'values': chunk_values,
                'bg_value': color_value}

    def get_format(self, chunk_values, h_start, v_start, h_stop, v_stop):
        return [[self._number_format]]

    def set_format(self, fmt):
        """Change display format"""
        # print(f"setting adapter format: {fmt}")
        self._number_format = fmt

    def from_clipboard_data_to_model_data(self, list_data):
        return list_data

    def get_axes_labels_and_data_values(self, row_min, row_max, col_min, col_max):
        axes_names = self.get_axes_area()
        axes_names = axes_names['values'] if isinstance(axes_names, dict) else axes_names
        hlabels = self.get_hlabels_values(col_min, col_max)
        vlabels = self.get_vlabels_values(row_min, row_max)
        raw_data = self.get_values(col_min, row_min, col_max, row_max)
        if isinstance(raw_data, list) and len(raw_data) == 0:
            raw_data = [[]]
        return axes_names, vlabels, hlabels, raw_data

    def move_axis(self, data, attributes, old_index, new_index):
        """Move an axis of the data array and associated attribute arrays.

        Parameters
        ----------
        data : array
            Array to transpose
        attributes : dict or None
            Dict of associated arrays.
        old_index: int
            Current index of axis to move.
        new_index: int
            New index of axis after transpose.

        Returns
        -------
        data : array
            Transposed input array
        attributes: dict
            Transposed associated arrays
        """
        raise NotImplementedError()

    def can_filter_axis(self, axis_idx) -> bool:
        return False

    def get_filter_names(self):
        """return [combo_label, ...]"""
        return []

    # TODO: change to get_filter_options(filter_idx, start, stop)
    #       ... in the end, filters will move to axes names
    #       AND possibly h/vlabels and
    #       must update the current command
    def get_filter_options(self, filter_idx) -> Optional[list]:
        """return [combo_values]"""
        return None

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
        global_changes = [
            CellValueChange(to_global(self.filtered_data.shape, self.current_filter, key),
                            old_value, new_value)
            for key, (old_value, new_value) in data_model_changes.items()
        ]
        return global_changes

    def get_sample(self):
        """Return a sample of the internal data"""
        # TODO: use default_buffer sizes instead, or, better yet, a new get_preferred_buffer_size() method
        height, width = self.shape2d()
        # TODO: use this instead (but it currently does not work because get_values does not always
        #       return a numpy array while the current code does
        # return self.get_values(0, 0, min(width, 20), min(height, 20))
        return self.get_data_values_and_attributes(0, 0, min(width, 20), min(height, 20))['values']

    def update_finite_min_max_values(self, finite_values: np.ndarray,
                                     h_start: int, v_start: int,
                                     h_stop: int, v_stop: int):
        """can return either two floats or two arrays"""

        # we need initial to support empty arrays
        vmin = np.nanmin(finite_values, initial=np.nan)
        vmax = np.nanmax(finite_values, initial=np.nan)

        self.vmin = (
            np.nanmin([self.vmin, vmin]) if self.vmin is not None else vmin)
        self.vmax = (
            np.nanmax([self.vmax, vmax]) if self.vmax is not None else vmax)
        return self.vmin, self.vmax

    def get_axes_area(self):
        # axes = self.filtered_data.axes
        # test axes.size == 0 is required in case an instance built as Array([]) is passed
        # test len(axes) == 0 is required when a user filters until getting a scalar (because in that case size is 1)
        # TODO: store this in the adapter
        # if axes.size == 0 or len(axes) == 0:
        #     return [[]]
        # else:
        shape = self.shape2d()
        row_idx_names = [name if name is not None else ''
                         for name in self.get_vnames()]
        num_v_axes = len(row_idx_names)
        col_idx_names = [name if name is not None else ''
                         for name in self.get_hnames()]
        num_h_axes = len(col_idx_names)
        if (not len(row_idx_names) and not len(col_idx_names)) or any(d == 0 for d in shape):
            return [[]]
        names = np.full((max(num_h_axes, 1), max(num_v_axes, 1)), '', dtype=object)
        if len(row_idx_names) > 1:
            names[-1, :-1] = row_idx_names[:-1]
        if len(col_idx_names) > 1:
            names[:-1, -1] = col_idx_names[:-1]
        part1 = row_idx_names[-1] if row_idx_names else ''
        part2 = col_idx_names[-1] if col_idx_names else ''
        sep = '\\' if part1 and part2 else ''
        names[-1, -1] = f'{part1}{sep}{part2}'

        current_sort = self.get_current_sort()
        sorted_axes = {axis_idx: ascending for axis_idx, label_idx, ascending in current_sort
                       if label_idx == -1}
        decoration = np.full_like(names, '', dtype=object)
        ascending_to_decoration = {
            True: 'arrow_up',
            False: 'arrow_down',
        }
        decoration[-1, :-1] = [ascending_to_decoration[sorted_axes[i]] if i in sorted_axes else ''
                               for i in range(len(row_idx_names) - 1)]
        return {'values': names.tolist(), 'decoration': decoration.tolist()}

    def get_current_sort(self) -> list[tuple]:
        """Return current sort

        Must be a list of tuples of the form
        (axis_idx, label_idx, ascending) where
        * axis_idx: is the index of the axis (of the label)
                    being sorted
        * label_idx: is the index of the label being sorted,
                     or -1 if the sort is by the axis labels themselves
        * ascending: bool

        Note that unsorted axes are not mentioned.
        """
        return self._current_sort

    # Adapter classes *may* implement this if can_sort_axis returns True for any axis
    def axis_sort_direction(self, axis_idx):
        """must return 'ascending', 'descending' or 'unsorted'"""
        for cur_sort_axis_idx, label_idx, ascending in self._current_sort:
            if cur_sort_axis_idx == axis_idx:
                return 'ascending' if ascending else 'descending'
        return 'unsorted'

    def hlabel_sort_direction(self, row_idx, col_idx):
        """must return 'ascending', 'descending' or 'unsorted'"""
        cell_axis_idx = self.hlabel_row_to_axis_num(row_idx)
        for axis_idx, label_idx, ascending in self._current_sort:
            if axis_idx == cell_axis_idx and label_idx == col_idx:
                return 'ascending' if ascending else 'descending'
        return 'unsorted'

    def can_filter_hlabel(self, row_idx, col_idx) -> bool:
        return False

    def can_sort_axis_labels(self, axis_idx) -> bool:
        return False

    def sort_axis_labels(self, axis_idx, ascending):
        pass

    # TODO: unsure a different result per label is useful. Per axis would probably be enough
    def can_sort_hlabel(self, row_idx, col_idx) -> bool:
        return False

    def sort_hlabel(self, row_idx, col_idx, ascending):
        pass

    def get_vlabels(self, start, stop) -> dict:
        chunk_values = self.get_vlabels_values(start, stop)
        if isinstance(chunk_values, list) and len(chunk_values) == 0:
            chunk_values = [[]]
        return {'values': chunk_values}

    def get_vlabels_values(self, start, stop):
        # Note that using some kind of lazy object here is pointless given that
        # we will use most of it (the buffer should not be much larger than the
        # viewport). It would make sense to define one big lazy object as
        # self._vlabels = Product([range(len(data))]) and use return
        # self._vlabels[start:stop] here but I am unsure it is worth it because
        # that would be slower than what we have now.
        return [[i] for i in range(start, stop)]

    def get_hlabels(self, start, stop):
        values = self.get_hlabels_values(start, stop)
        return {'values': values, 'decoration': self.get_hlabels_decorations(start, stop, values)}

    def get_hlabels_values(self, start, stop):
        return [list(range(start, stop))]

    def hlabel_row_to_axis_num(self, row_idx):
        return row_idx + self.num_v_axes()

    def num_v_axes(self):
        return 1

    def get_hlabels_decorations(self, start, stop, labels):
        current_sort = self.get_current_sort()
        sorted_labels_by_axis = {}
        for axis_idx, label_idx, ascending in current_sort:
            sorted_labels_by_axis.setdefault(axis_idx, {})[label_idx] = ascending
        ascending_to_decoration = {
            True: 'arrow_up',
            False: 'arrow_down',
        }
        decorations = []
        for row_idx in range(len(labels)):
            row_axis_idx = self.hlabel_row_to_axis_num(row_idx)
            axis_sorted_labels = sorted_labels_by_axis.get(row_axis_idx, {})
            decoration_row = [
                ascending_to_decoration[axis_sorted_labels[col_idx]] if col_idx in axis_sorted_labels else ''
                for col_idx in range(start, stop)
            ]
            decorations.append(decoration_row)
        return decorations

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

    # FIXME (unsure this is still the case): this function does not support None axes_names, vlabels and hlabels
    #        which _selection_data() produces in some cases (notably when working
    #        on a scalar array). Unsure if we should fix _selection_data or this
    #        method though.
    def get_combined_values(self, row_min, row_max, col_min, col_max):
        """Return ...

        Parameters
        ----------

        Returns
        -------
        list of list of built-in Python scalars (None, bool, int, float, str)
        """
        axes_names, vlabels, hlabels, raw_data = (
            self.get_axes_labels_and_data_values(row_min, row_max,
                                                 col_min, col_max)
        )
        return self.combine_labels_and_data(raw_data, axes_names,
                                            vlabels, hlabels)

    def to_string(self, row_min, row_max, col_min, col_max, sep='\t'):
        """Copy selection as tab-separated (clipboard) text

        Returns
        -------
        str
        """
        data = self.get_combined_values(row_min, row_max, col_min, col_max)

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

        data = self.get_combined_values(row_min, row_max, col_min, col_max)
        # convert (row) generators to lists then array
        # TODO: the conversion to array is currently necessary even though xlwings will translate it back to a list
        #       anyway. The problem is that our lists contains numpy types and especially np.str_ crashes xlwings.
        #       unsure how we should fix this properly: in xlwings, or change get_combined_data() to return only
        #       standard Python types.
        array = np.array([list(r) for r in data])

        # Create a new Excel instance. We cannot simply use xw.view(array)
        # because it reuses the active Excel instance if any, and if that one
        # is hidden, the user will not see anything
        app = xw.App(visible=True)

        # Activate XLA(M) addins. By default, they are not activated when an
        # Excel Workbook is opened via COM
        xl_app = app.api
        for i in range(1, xl_app.AddIns.Count + 1):
            addin = xl_app.AddIns(i)
            addin_path = addin.FullName
            if addin.Installed and '.xll' not in addin_path.lower():
                xl_app.Workbooks.Open(addin_path)

        # Dump array to first sheet
        book = app.books[0]
        sheet = book.sheets[0]
        with app.properties(screen_updating=False):
            sheet["A1"].value = array
            # Unsure whether we should do this or not
            # sheet.tables.add(sheet["A1"].expand())
            sheet.autofit()

        # Move Excel Window at the front. Without steal_focus it does not seem
        # to do anything
        app.activate(steal_focus=True)

    def plot(self, row_min, row_max, col_min, col_max):
        """Return a matplotlib.Figure object for selected subset.

        Returns
        -------
        A matplotlib.Figure object.
        """
        from matplotlib.figure import Figure

        # we do not use the axes_names part because the position of axes names is up to the adapter
        _, vlabels, hlabels, raw_data = self.get_axes_labels_and_data_values(row_min, row_max, col_min, col_max)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"AbstractAdapter.plot {vlabels=} {hlabels=}")
            logger.debug(f"{raw_data=}")
        if not isinstance(raw_data, np.ndarray):
            # Without dtype=object, in the presence of a string, raw_data will
            # be entirely converted to strings which is not what we want.
            raw_data = np.asarray(raw_data, dtype=object)
            raw_data = raw_data.reshape((raw_data.shape[0], -1))
        assert isinstance(raw_data, np.ndarray), f"got data of type {type(raw_data)}"
        assert raw_data.ndim == 2, f"ndim is {raw_data.ndim}"
        finite_values = get_finite_numeric_values(raw_data)
        figure = Figure()

        # create an axis
        ax = figure.add_subplot()

        # we have a list of rows but we want a list of columns
        xlabels = list(zip(*hlabels))
        ylabels = vlabels

        xlabel = self.get_hname()
        ylabel = self.get_vname()

        height, width = finite_values.shape
        if width == 1:
            # plot one column
            xlabels, ylabels = ylabels, xlabels
            xlabel, ylabel = ylabel, xlabel
            height, width = width, height
            finite_values = finite_values.T

        # plot each row as a line
        xticklabels = ['\n'.join(str(label) for label in label_col)
                       for label_col in xlabels]
        xdata = np.arange(width)
        for data_row, ylabels_row in zip(finite_values, ylabels):
            row_label = ' '.join(str(label) for label in ylabels_row)
            ax.plot(xdata, data_row, label=row_label)

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


class AbstractColumnarAdapter(AbstractAdapter):
    """For adapters where color is per column"""

    def __init__(self, data, attributes=None):
        super().__init__(data, attributes)
        self.vmin = {}
        self.vmax = {}

    def update_finite_min_max_values(self, finite_values: np.ndarray, 
                                     h_start: int, v_start: int,
                                     h_stop: int, v_stop: int):

        assert isinstance(self.vmin, dict) and isinstance(self.vmax, dict)
        assert h_stop >= h_start

        # per column => axis=0
        local_vmin = np.nanmin(finite_values, axis=0, initial=np.nan)
        local_vmax = np.nanmax(finite_values, axis=0, initial=np.nan)
        num_cols = h_stop - h_start
        assert local_vmin.shape == (num_cols,), \
            (f"unexpected shape: {local_vmin.shape} ({finite_values.shape=}) vs "
             f"{(num_cols,)} ({h_start=} {h_stop=})")
        # vmin or self.vmin can both be nan (if the whole section data
        # is/was nan)
        global_vmin = self.vmin
        global_vmax = self.vmax
        vmin_slice = np.empty(num_cols, dtype=np.float64)
        vmax_slice = np.empty(num_cols, dtype=np.float64)
        for global_col_idx in range(h_start, h_stop):
            local_col_idx = global_col_idx - h_start

            col_min = np.nanmin([global_vmin.get(global_col_idx, np.nan),
                                 local_vmin[local_col_idx]])
            # update the global vmin dict inplace
            global_vmin[global_col_idx] = col_min
            vmin_slice[local_col_idx] = col_min

            col_max = np.nanmax([global_vmax.get(global_col_idx, np.nan),
                                 local_vmax[local_col_idx]])
            # update the global vmax dict inplace
            global_vmax[global_col_idx] = col_max
            vmax_slice[local_col_idx] = col_max
        return vmin_slice, vmax_slice


# this is NOT (and should not inherit from) AbstractAdapter
# instances of this class "adapt" a Path object with specific suffixes to a type understood/handled by a real
# (arrayish) adapter (ie a descendant from AbstractAdapter)
class AbstractPathAdapter:
    # Path adapters MAY override this method, but should probably override
    # open() instead
    @classmethod
    def get_file_handler_and_adapter_creator(cls, fpath):
        file_handler_object = cls.open(fpath)
        return file_handler_object, get_adapter_creator(file_handler_object)

    # Path adapters MAY override these methods
    @classmethod
    def open(cls, fpath):
        """The result of this method will be stored in the .data
        attribute and passed as argument to the adapter class"""
        raise NotImplementedError()


class DirectoryPathAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        # taking absolute() allows going outside of the initial directory
        # via double click. This is both good and bad.
        data = data.absolute()
        super().__init__(data=data, attributes=attributes)
        path_objs = list(data.iterdir())
        # sort by type then name
        path_objs.sort(key=lambda p: (not p.is_dir(), p.name))
        parent_dir = data.parent
        if parent_dir != data:
            path_objs.insert(0, parent_dir)
        self._path_objs = path_objs

        def file_mtime_as_str(p) -> str:
            try:
                mt_time = datetime.fromtimestamp(p.stat().st_mtime)
                return mt_time.strftime('%d/%m/%Y %H:%M')
            except Exception:
                return ''

        self._list = [(
            p.name if p != parent_dir else '..',
            # give the mtime of the "current" directory
           file_mtime_as_str(p if p != parent_dir else data),
           '<directory>' if p.is_dir() else p.stat().st_size
        ) for p in path_objs]
        self._colnames = ['Name', 'Time Modified', 'Size']

    def shape2d(self):
        return len(self._list), len(self._colnames)

    def get_hlabels_values(self, start, stop):
        return [self._colnames[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return [row[h_start:h_stop]
                for row in self._list[v_start:v_stop]]

    def cell_activated(self, row_idx, column_idx):
        return self._path_objs[row_idx].absolute()


@adapter_for('pathlib.Path')
def get_path_suffix_adapter(fpath):
    logger.debug(f"get_path_suffix_adapter('{fpath}')")
    if fpath.suffix.lower() in PATH_SUFFIX_ADAPTERS:
        path_adapter_cls, required_module = PATH_SUFFIX_ADAPTERS[fpath.suffix]
        if required_module is not None:
            if required_module not in sys.modules:
                import importlib
                try:
                    importlib.import_module(required_module)
                except ImportError:
                    logger.warn(f"Failed to import '{required_module}' module, "
                                f"which is required to handle {fpath.suffix} "
                                f"files")
                    return None
        return path_adapter_cls
    elif fpath.is_dir():
        return DirectoryPathAdapter
    else:
        return None


class SequenceAdapter(AbstractAdapter):
    def __init__(self, data, attributes):
        super().__init__(data, attributes)
        self.sorted_data = data
        self.sorted_indices = range(len(data))

    def shape2d(self):
        return len(self.data), 1

    def get_vnames(self):
        return ['index']

    def get_hlabels_values(self, start, stop):
        return [['value']]

    def get_vlabels_values(self, start, stop):
        # Note that using some kind of lazy object here is pointless given that
        # we will use most of it (the buffer should not be much larger than the
        # viewport). It would make sense to define one big lazy object as
        # self._vlabels = Product([range(len(data))]) and use return
        # self._vlabels[start:stop] here but I am unsure it is worth it because
        # that would be slower than what we have now.
        return [[i] for i in self.sorted_indices[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        chunk_values = self.sorted_data[v_start:v_stop]
        # use a numpy array to avoid confusing the model if some
        # elements are sequence themselves
        array_values = np.empty((len(chunk_values), 1), dtype=object)
        array_values[:, 0] = chunk_values
        return array_values

    def can_sort_hlabel(self, row_idx, col_idx):
        return True

    def sort_hlabel(self, row_idx, col_idx, ascending):
        self._current_sort = [(self.num_v_axes() + row_idx, col_idx, ascending)]
        self.sorted_indices = indirect_sort(self.data, ascending)
        self.sorted_data = sorted(self.data, reverse=not ascending)


class NamedTupleAdapter(AbstractAdapter):
    def shape2d(self):
        return len(self.data), 1

    def get_vnames(self):
        return ['attribute']

    def get_hlabels_values(self, start, stop):
        return [['value']]

    def get_vlabels_values(self, start, stop):
        return [[k] for k in self.data._fields[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return [[v] for v in self.data[v_start:v_stop]]


@adapter_for(collections.abc.Sequence)
def get_sequence_adapter(data):
    namedtuple_attrs = ['_asdict', '_field_defaults', '_fields', '_make', '_replace']
    # Named tuples have no special parent class, so we cannot dispatch using the type
    # of data and need to check the presence of NamedTuple specific attributes instead
    if isinstance(data, str):
        return None
    elif all(hasattr(data, attr) for attr in namedtuple_attrs):
        return NamedTupleAdapter
    else:
        return SequenceAdapter


@adapter_for(collections.abc.Mapping)
class MappingAdapter(AbstractAdapter):
    def shape2d(self):
        return len(self.data), 1

    def get_vnames(self):
        return ['key']

    def get_hlabels_values(self, start, stop):
        return [['value']]

    def get_vlabels_values(self, start, stop):
        # using islice instead of caching list(data.keys()) and list(data.values()) in __init__
        # make things *much* faster to display the first elements of very large dicts at
        # the expense of making the display of the last elements about twice as slow.
        # It seems a desirable tradeoff, especially given the lower memory usage and
        # the absence of stale cache problem. Performance-wise, we could cache keys() and
        # values() here (instead of in __init__) if start or stop is above some threshold
        # but I am unsure it is worth the added complexity.
        return [[k] for k in itertools.islice(self.data.keys(), start, stop)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        values_chunk = itertools.islice(self.data.values(), v_start, v_stop)
        return [[v] for v in values_chunk]


# @adapter_for(object)
# class ObjectAdapter(AbstractAdapter):
#     def __init__(self, data, attributes):
#         super().__init__(data=data, attributes=attributes)
#         self._fields = [k for k in dir(data) if not k.startswith('_') and type(getattr(data, k)) not in
#                         {types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType}]
#
#     def shape2d(self):
#         return len(self._fields), 1
#
#     def get_vnames(self):
#         return ['key']
#
#     def get_hlabels(self, start, stop):
#         return [['value']]
#
#     def get_vlabels(self, start, stop):
#         return [[f] for f in self._fields[start:stop]]
#
#     def get_values(self, h_start, v_start, h_stop, v_stop):
#         return [[getattr(self.data, k)] for k in self._fields[v_start:v_stop]]


@adapter_for(collections.abc.Collection)
class CollectionAdapter(AbstractAdapter):
    def shape2d(self):
        return len(self.data), 1

    def get_hlabels_values(self, start, stop):
        return [['value']]

    def get_vlabels_values(self, start, stop):
        return [[''] for i in range(start, stop)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return [[v] for v in itertools.islice(self.data, v_start, v_stop)]


# Specific adapter just to change the label
@adapter_for(collections.abc.KeysView)
class KeysViewAdapter(CollectionAdapter):
    def get_hlabels_values(self, start, stop):
        return [['key']]


@adapter_for(collections.abc.ItemsView)
class ItemsViewAdapter(CollectionAdapter):
    def shape2d(self):
        return len(self.data), 2

    def get_hlabels_values(self, start, stop):
        return [['key', 'value']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        # slicing self.data already returns "tuple" rows
        return list(itertools.islice(self.data, v_start, v_stop))


def get_finite_numeric_values(array: np.ndarray) -> np.ndarray:
    """return a copy of array with non numeric, -inf or inf values set to nan"""
    dtype = array.dtype
    finite_value = array
    # TODO: there are more complex dtypes than this. Is there a way to get them all in one shot?
    if dtype in (np.complex64, np.complex128):
        # for complex numbers, shading will be based on absolute value
        # FIXME: this is fine for coloring purposes but not for determining
        #        format (or plotting?)
        finite_value = np.abs(finite_value)
    elif dtype.type is np.object_:
        # change non numeric to nan
        finite_value = np.where(is_number_value_vectorized(finite_value),
                                finite_value,
                                np.nan)
        finite_value = finite_value.astype(np.float64)
    elif np.issubdtype(dtype, np.bool_):
        finite_value = finite_value.astype(np.int8)
    elif not np.issubdtype(dtype, np.number):
        # if the whole array is known to be non numeric, we do not need
        # to compute anything
        return np.full(array.shape, np.nan, dtype=np.float64)

    assert np.issubdtype(finite_value.dtype, np.number)

    # change inf and -inf to nan (setting them to 0 or to very large numbers is
    # not an option because it would "dampen" normal values)
    return np.where(np.isfinite(finite_value), finite_value, np.nan)


# only used in LArray adapter. it should use the same code path as the rest
# though
def get_color_value(array, global_vmin, global_vmax, axis=None):
    assert isinstance(array, np.ndarray)
    try:
        finite_value = get_finite_numeric_values(array)

        vmin = np.nanmin(finite_value, axis=axis)
        if global_vmin is not None:
            # vmin or global_vmin can both be nan (if the whole section data is/was nan)
            global_vmin = np.nanmin([global_vmin, vmin], axis=axis)
        else:
            global_vmin = vmin
        vmax = np.nanmax(finite_value, axis=axis)
        if global_vmax is not None:
            # vmax or global_vmax can both be nan (if the whole section data is/was nan)
            global_vmax = np.nanmax([global_vmax, vmax], axis=axis)
        else:
            global_vmax = vmax
        color_value = scale_to_01range(finite_value, global_vmin, global_vmax)
    except (ValueError, TypeError):
        global_vmin = None
        global_vmax = None
        color_value = None
    return color_value, global_vmin, global_vmax


class NumpyHomogeneousArrayAdapter(AbstractAdapter):
    def shape2d(self):
        return nd_shape_to_2d(self.data.shape, num_h_axes=1)

    def get_vnames(self):
        return ['' for axis_len in self.data.shape[:-1]]

    def get_hlabels_values(self, start, stop):
        if self.data.ndim > 0:
            return [list(range(start, stop))]
        else:
            return [['']]

    def get_vlabels_values(self, start, stop):
        if self.data.ndim > 0:
            vlabels = Product([range(axis_len) for axis_len in self.data.shape[:-1]])
            return vlabels[start:stop]
        else:
            return [['']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        data2d = self.data.reshape(nd_shape_to_2d(self.data.shape))
        return data2d[v_start:v_stop, h_start:h_stop]


class NumpyStructuredArrayAdapter(AbstractColumnarAdapter):
    def shape2d(self):
        shape = self.data.shape + (len(self.data.dtype.names),)
        return nd_shape_to_2d(shape, num_h_axes=1)

    def get_vnames(self):
        return ['' for axis_len in self.data.shape]

    def get_hlabels_values(self, start, stop):
        return [list(self.data.dtype.names[start:stop])]

    def get_vlabels_values(self, start, stop):
        vlabels = Product([range(axis_len) for axis_len in self.data.shape])
        return vlabels[start:stop]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        # TODO: this works nicely but isn't any better for users because number of decimals
        #       is not auto-detected and cannot be changed. I think I could implement
        #       auto-detection *relatively* easily but at this point I don't know
        #       how to implement changing it.
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
def get_np_array_adapter(data):
    if data.dtype.names is not None:
        return NumpyStructuredArrayAdapter
    else:
        return NumpyHomogeneousArrayAdapter


class MemoryViewAdapter(NumpyHomogeneousArrayAdapter):
    def __init__(self, data, attributes):
        # no data copy is necessary for converting memoryview <-> numpy array
        # and a memoryview >1D cannot be sliced (only indexed with a single
        # element) so it is much easier *and* efficient to convert to a numpy
        # array and display that
        super().__init__(np.asarray(data), attributes)


@adapter_for(memoryview)
def get_mv_array_adapter(data):
    if len(data.format) > 1:
        # memoryview with 'structured' formats cannot be indexed
        return None
    else:
        return MemoryViewAdapter


@adapter_for(la.Array)
class LArrayArrayAdapter(AbstractAdapter):
    num_axes_to_display_horizontally = 1

    def __init__(self, data, attributes):
        # self.num_axes_to_display_horizontally = min(data.ndim, 2)
        data = la.asarray(data)
        if attributes is not None:
            attributes = {k: la.asarray(v) for k, v in attributes.items()}
        super().__init__(data, attributes)
        # TODO: should not be needed (this is only used in ArrayDelegate)
        self.dtype = data.dtype

        self.filtered_data = self.data
        self.filtered_attributes = self.attributes
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
            # It is not a problem if the target labels (which we do not know
            # here) do not match with the source labels that we are stripping
            # because pasting values to the exact same cells we copied them
            # from has little interest
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
        return nd_shape_to_2d(self.filtered_data.shape,
                              num_h_axes=self.num_axes_to_display_horizontally)

    def can_filter_axis(self, axis_idx) -> bool:
        return True

    def get_filter_names(self):
        return self.data.axes.display_names

    def get_filter_options(self, filter_idx):
        return self.data.axes[filter_idx].labels

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
        cur_filter = self.current_filter
        axis = self.data.axes[filter_idx]
        if not indices or len(indices) == len(axis):
            if filter_idx in cur_filter:
                del cur_filter[filter_idx]
        else:
            if len(indices) == 1:
                cur_filter[filter_idx] = indices[0]
            else:
                cur_filter[filter_idx] = indices

        # cur_filter is a {axis_idx: axis_indices} dict
        # full_indices_filter is a tuple
        full_indices_filter = tuple(
            cur_filter[axis_idx] if axis_idx in cur_filter else slice(None)
            for axis_idx in range(len(self.data.axes))
        )
        self.filtered_data = self._filter_data(self.data, full_indices_filter)
        if self.attributes is not None:
            self.filtered_attributes = {k: self._filter_data(v, full_indices_filter)
                                        for k, v in self.attributes.items()}

    def can_sort_hlabel(self, row_idx, col_idx):
        return self.filtered_data.ndim == 2

    def sort_hlabel(self, row_idx, col_idx, ascending):
        self._current_sort = [(self.num_v_axes() + row_idx, col_idx, ascending)]
        arr = self.filtered_data
        assert arr.ndim == 2
        row_axis = arr.axes[0]
        col_axis = arr.axes[-1]
        key = col_axis.i[col_idx]
        sort_indices = arr[key].indicesofsorted(ascending=ascending)
        assert sort_indices.ndim == 1
        indexer = row_axis.i[sort_indices.data]
        self.filtered_data = arr[indexer]
        if self.attributes is not None:
            self.filtered_attributes = {k: v[indexer]
                                        for k, v in self.filtered_attributes.items()}

    def can_sort_axis_labels(self, axis_idx) -> bool:
        return True

    def sort_axis_labels(self, axis_idx, ascending):
        self._current_sort = [(axis_idx, -1, ascending)]
        assert isinstance(self.filtered_data, la.Array)
        self.filtered_data = self.filtered_data.sort_labels(axis_idx, ascending=ascending)
        if self.attributes is not None:
            self.filtered_attributes = {k: v.sort_labels(axis_idx, ascending=ascending)
                                        for k, v in self.filtered_attributes.items()}

    def get_data_values_and_attributes(self, h_start, v_start, h_stop, v_stop):
        # data
        # ====
        chunk_values = self.get_values(h_start, v_start, h_stop, v_stop)
        chunk_data = {
            'editable': [[True]],
            'data_format': [[self._number_format]],
            'values': chunk_values
        }

        # user-defined attributes (e.g. user-provided bg_value)
        # =======================
        if self.filtered_attributes is not None:
            chunk_data.update({k: self.get_chunk_from_data(v, h_start, v_start, h_stop, v_stop)
                               for k, v in self.filtered_attributes.items()})
        # we are not doing this above like for editable and data_format for performance reasons
        # when bg_value is a user-provided value (and we do not need a computed one)
        if 'bg_value' not in chunk_data:
            # "default" bg_value computed on the subset asked by the model
            bg_value, self.vmin, self.vmax = get_color_value(chunk_values, self.vmin, self.vmax)
            chunk_data['bg_value'] = bg_value
        return chunk_data

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self.get_chunk_from_data(self.filtered_data,
                                        h_start, v_start,
                                        h_stop, v_stop)

    def get_chunk_from_data(self, data, h_start, v_start, h_stop, v_stop):
        # get filtered data as Numpy 2D array
        assert isinstance(data, la.Array)
        np_data = data.data
        assert isinstance(np_data, np.ndarray)
        shape2d = nd_shape_to_2d(np_data.shape, self.num_axes_to_display_horizontally)
        raw_data = np_data.reshape(shape2d)
        return raw_data[v_start:v_stop, h_start:h_stop]

    def get_vnames(self):
        axes = self.filtered_data.axes
        num_v_axes = max(len(axes) - self.num_axes_to_display_horizontally, 0)
        return axes.display_names[:num_v_axes]

    def get_hnames(self):
        axes = self.filtered_data.axes
        num_v_axes = max(len(axes) - self.num_axes_to_display_horizontally, 0)
        return axes.display_names[num_v_axes:]

    def get_vlabels_values(self, start, stop):
        axes = self.filtered_data.axes
        # test data.size == 0 is required in case an instance built as Array([]) is passed
        # test len(axes) == 0 is required when a user filters until getting a scalar (because in that case size is 1)
        # TODO: store this in the adapter
        if axes.size == 0 or len(axes) == 0:
            return [[]]
        elif len(axes) <= self.num_axes_to_display_horizontally:
            # all axes are horizontal => a single empty vlabel
            return [['']]
        else:
            # we must not convert the *whole* axes to raw python objects here (e.g. using tolist) because this would be
            # too slow for huge axes
            v_axes = axes[:-self.num_axes_to_display_horizontally] \
                if self.num_axes_to_display_horizontally else axes
            # CHECK: store self._vlabels in adapter?
            vlabels = Product([axis.labels for axis in v_axes])
            return vlabels[start:stop]

    def get_hlabels_values(self, start, stop):
        axes = self.filtered_data.axes
        # test data.size == 0 is required in case an instance built as Array([]) is passed
        # test len(axes) == 0 is required when a user filters until to get a scalar
        # TODO: store this in the adapter
        if axes.size == 0 or len(axes) == 0:
            return [[]]
        elif not self.num_axes_to_display_horizontally:
            # all axes are vertical => a single empty hlabel
            return [['']]
        else:
            haxes = axes[-self.num_axes_to_display_horizontally:]
            hlabels = Product([axis.labels for axis in haxes])
            section_labels = hlabels[start:stop]
            # we have a list of columns but we need a list of rows
            return [[label_col[row_num] for label_col in section_labels]
                    for row_num in range(self.num_axes_to_display_horizontally)]

    def get_sample(self):
        """Return a sample of the internal data"""
        np_data = self.filtered_data.data
        # this will yield a data sample of max 200
        return get_sample(np_data, 200)

    def move_axis(self, data, attributes, old_index, new_index):
        assert isinstance(data, la.Array)
        new_axes = data.axes.copy()
        new_axes.insert(new_index, new_axes.pop(new_axes[old_index]))
        data = data.transpose(new_axes)
        if attributes is not None:
            assert isinstance(attributes, dict)
            attributes = {k: v.transpose(new_axes) for k, v in attributes.items()}
        return data, attributes

    # TODO: move this to a DenseArrayAdapter superclass
    def map_filtered_to_global(self, filtered_shape, filter, local2dkey):
        """
        transform local (filtered) 2D (row_idx, col_idx) key to global (unfiltered) ND key
        (axis0_pos, axis1_pos, ..., axisN_pos). This is positional only (no labels).
        """
        row, col = local2dkey

        localndkey = list(np.unravel_index(row, filtered_shape[:-1])) + [col]

        # add the "scalar" parts of the filter to it (ie the parts of the filter which removed dimensions)
        scalar_filter_keys = [axis_idx for axis_idx, axis_filter in filter.items()
                              if np.isscalar(axis_filter)]
        for axis_idx in sorted(scalar_filter_keys):
            localndkey.insert(axis_idx, filter[axis_idx])

        # translate local to global for filtered dimensions which are still present (non scalar)
        return tuple(
            axis_pos if axis_idx not in filter or np.isscalar(filter[axis_idx]) else filter[axis_idx][axis_pos]
            for axis_idx, axis_pos in enumerate(localndkey)
        )


# cannot let the default Sequence adapter be used because axis[slice] is an
# LGroup
@adapter_for(la.Axis)
class LArrayAxisAdapter(NumpyHomogeneousArrayAdapter):
    def __init__(self, data, attributes):
        super().__init__(data.labels, attributes)


@adapter_for('array.array')
class ArrayArrayAdapter(AbstractAdapter):
    def shape2d(self):
        return len(self.data), 1

    def get_hlabels_values(self, start, stop):
        return [['']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return [[v] for v in self.data[v_start:v_stop]]


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
    def __init__(self, data, attributes):
        super().__init__(data, attributes)
        self._sheet_names = data.sheet_names()

    def get_hlabels_values(self, start, stop):
        return [['sheet name']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return [[sheet_name]
                for sheet_name in self._sheet_names[v_start:v_stop]]

    def cell_activated(self, row_idx, column_idx):
        return self.data[row_idx]


@adapter_for('larray.inout.xw_excel.Sheet')
class SheetAdapter(AbstractAdapter):
    def shape2d(self):
        return self.data.shape

    def get_hlabels_values(self, start, stop):
        return [[excel_colname(i) for i in range(start, stop)]]

    def get_vlabels_values(self, start, stop):
        # +1 because excel rows are 1 based
        return [[i] for i in range(start + 1, stop + 1)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self.data[v_start:v_stop, h_start:h_stop].__array__()


@adapter_for('larray.inout.xw_excel.Range')
class RangeAdapter(AbstractAdapter):
    def shape2d(self):
        return self.data.shape

    def get_hlabels_values(self, start, stop):
        # - 1 because data.column is 1-based (Excel) while excel_colname is 0-based
        offset = self.data.column - 1
        return [[excel_colname(i) for i in range(offset + start, offset + stop)]]

    def get_vlabels_values(self, start, stop):
        offset = self.data.row
        return [[i] for i in range(offset + start, offset + stop)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self.data[v_start:v_stop, h_start:h_stop].__array__()


@adapter_for('pandas.DataFrame')
class PandasDataFrameAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        pd = sys.modules['pandas']
        assert isinstance(data, pd.DataFrame)
        super().__init__(data, attributes=attributes)
        self.sorted_data = data

    def shape2d(self):
        return self.data.shape

    def get_filter_names(self):
        return []
        # pd = sys.modules['pandas']
        # df = self.data
        # assert isinstance(df, pd.DataFrame)
        # def get_index_names(index):
        #     if isinstance(index, pd.MultiIndex):
        #         return list(index.names)
        #     else:
        #         return [index.name if index.name is not None else '']
        # return get_index_names(df.index) + get_index_names(df.columns)

    def num_v_axes(self):
        return self.data.index.nlevels

    def get_filter_options(self, filter_idx):
        pd = sys.modules['pandas']
        df = self.data
        assert isinstance(df, pd.DataFrame)
        def get_values(index):
            if isinstance(index, pd.MultiIndex):
                return list(index.levels)
            else:
                return [index.values]
        # FIXME: this is awfully inefficient
        all_filters_options = get_values(df.index) + get_values(df.columns)
        return all_filters_options[filter_idx]

    def filter_data(self, data, filter):
        """
        filter is a {axis_idx: axis_indices} dict
        """
        if data is None or filter is None:
            return data

        pd = sys.modules['pandas']
        assert isinstance(data, pd.DataFrame)
        if isinstance(data.index, pd.MultiIndex) or isinstance(data.columns, pd.MultiIndex):
            print("WARNING: filtering with ndim > 2 not implemented yet")
            return data

        indexer = tuple(filter.get(axis_idx, slice(None))
                        for axis_idx in range(self.data.ndim))
        res = data.iloc[indexer]
        if isinstance(res, pd.Series):
            res = res.to_frame()
        return res

    def get_hnames(self):
        return self.data.columns.names

    def get_vnames(self):
        return self.data.index.names

    def get_vlabels_values(self, start, stop):
        pd = sys.modules['pandas']
        index = self.sorted_data.index[start:stop]
        if isinstance(index, pd.MultiIndex):
            return [list(row) for row in index.values]
        else:
            return index.values[:, np.newaxis]

    def get_hlabels_values(self, start, stop):
        pd = sys.modules['pandas']
        index = self.sorted_data.columns[start:stop]
        if isinstance(index, pd.MultiIndex):
            return [index.get_level_values(i).values
                    for i in range(index.nlevels)]
        else:
            return [index.values]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self.sorted_data.iloc[v_start:v_stop, h_start:h_stop].values

    def can_sort_hlabel(self, row_idx, col_idx):
        # allow sorting on columns but not rows
        return True

    def sort_hlabel(self, row_idx, col_idx, ascending):
        self._current_sort = [(self.num_v_axes() + row_idx, col_idx, ascending)]
        self.sorted_data = self.data.sort_values(self.data.columns[col_idx], ascending=ascending)


@adapter_for('pandas.Series')
class PandasSeriesAdapter(AbstractAdapter):
    def __init__(self, data, attributes):
        pd = sys.modules['pandas']
        assert isinstance(data, pd.Series)
        super().__init__(data=data, attributes=attributes)

    def shape2d(self):
        return len(self.data), 1

    def get_vnames(self):
        return self.data.index.names

    def get_vlabels_values(self, start, stop):
        pd = sys.modules['pandas']
        index = self.data.index[start:stop]
        if isinstance(index, pd.MultiIndex):
            # returns a 1D array of tuples
            return index.values
        else:
            return index.values[:, np.newaxis]

    def get_hlabels_values(self, start, stop):
        return [['']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        assert h_start == 0
        return self.data.iloc[v_start:v_stop].values.reshape(-1, 1)


@adapter_for('pyarrow.Array')
class PyArrowArrayAdapter(AbstractAdapter):
    def shape2d(self):
        return len(self.data), 1

    def get_hlabels_values(self, start, stop):
        return [['value']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self.data[v_start:v_stop].to_numpy(zero_copy_only=False).reshape(-1, 1)


# Contrary to other Path adapters, this one is both a File *and* Path adapter
# because it is more efficient to NOT keep the file open (because the pyarrow
# API only allows limiting which columns are read when opening the file)
@path_adapter_for('.feather', 'pyarrow.ipc')
@path_adapter_for('.ipc', 'pyarrow.ipc')
@path_adapter_for('.arrow', 'pyarrow.ipc')
@adapter_for('pyarrow.RecordBatchFileReader')
class FeatherFileAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        if isinstance(data, str):
            data = Path(data)
        # assert isinstance(data, (Path, pyarrow.RecordBatchFileReader))
        super().__init__(data=data, attributes=attributes)

        # TODO: take pandas metadata index columns into account:
        # - display those columns as labels
        # - remove those columns from shape
        # - do not read those columns in get_values
        with self._open_file() as f:
            self._colnames = f.schema.names
            self._num_columns = len(f.schema)
            self._num_record_batches = f.num_record_batches

        self._batch_nrows = np.zeros(self._num_record_batches, dtype=np.int64)
        maxint = np.iinfo(np.int64).max
        self._batch_ends = np.full(self._num_record_batches, maxint, dtype=np.int64)
        # TODO: get this from somewhere else
        default_buffer_rows = 40
        self._num_batches_indexed = 0
        self._num_rows = None
        self._index_up_to(default_buffer_rows)

    def _open_file(self, col_indices=None):
        """col_indices is only taken into account if self.data is a Path"""
        ipc = sys.modules['pyarrow.ipc']
        if isinstance(self.data, Path):
            if col_indices is not None:
                options = ipc.IpcReadOptions(included_fields=col_indices)
            else:
                options = None
            return ipc.open_file(self.data, options=options)
        else:
            assert isinstance(self.data, ipc.RecordBatchFileReader)
            return self.data

    def _get_batches(self, start_batch, stop_batch, col_indices: list[int]) -> list:
        """stop_batch is not included"""
        logger.debug(f"FeatherFileAdapter._get_batches({start_batch}, "
                     f"{stop_batch}, {col_indices})")
        batch_indices = range(start_batch, stop_batch)
        if isinstance(self.data, Path):
            with self._open_file(col_indices=col_indices) as f:
                return [f.get_batch(i) for i in batch_indices]
        else:
            return [self.data.get_batch(i).select(col_indices)
                    for i in batch_indices]

    def shape2d(self):
        nrows = self._num_rows if self._num_rows is not None else self._estimated_num_rows
        return nrows, self._num_columns

    def get_hlabels_values(self, start, stop):
        return [self._colnames[start:stop]]

    def _index_up_to(self, num_rows):
        if self._num_batches_indexed == 0:
            last_indexed_batch_end = 0
        else:
            last_indexed_batch_end = self._batch_ends[self._num_batches_indexed - 1]

        if num_rows <= last_indexed_batch_end:
            return

        with self._open_file(col_indices=[0]) as f:
            while (num_rows > last_indexed_batch_end and
                   self._num_batches_indexed < self._num_record_batches):
                batch_num = self._num_batches_indexed
                batch_rows = f.get_batch(batch_num).num_rows
                last_indexed_batch_end += batch_rows
                self._batch_nrows[batch_num] = batch_rows
                self._batch_ends[batch_num] = last_indexed_batch_end
                self._num_batches_indexed += 1

            if self._num_batches_indexed == self._num_record_batches:
                # we are fully indexed
                self._num_rows = last_indexed_batch_end
                self._estimated_num_rows = None
            else:
                # we are not fully indexed
                self._num_rows = None
                # if we have not already indexed the last batch
                if self._batch_nrows[-1] == 0:
                    last_batch = self._num_record_batches - 1
                    self._batch_nrows[-1] = f.get_batch(last_batch).num_rows
                # we do not count the last batch which usually has a different length
                estimated_rows_per_batch = np.mean(self._batch_nrows[:self._num_batches_indexed])
                self._estimated_num_rows = int(estimated_rows_per_batch *
                                               (self._num_record_batches - 1)
                                               + self._batch_nrows[-1])

    def get_values(self, h_start, v_start, h_stop, v_stop):
        pyarrow = sys.modules['pyarrow']
        self._index_up_to(v_stop)
        # - 1 because the last row is not included
        start_batch, stop_batch = np.searchsorted(self._batch_ends,
                                                  v=[v_start, v_stop - 1],
                                                  side='right')
        # stop_batch is not included
        stop_batch += 1
        chunk_start = self._batch_ends[start_batch - 1] if start_batch > 0 else 0
        col_indices = list(range(h_start, h_stop))
        batches = self._get_batches(start_batch, stop_batch, col_indices)
        if len(batches) > 1:
            combined = pyarrow.concat_batches(batches)
        else:
            combined = batches[0]
        return combined[v_start - chunk_start:v_stop - chunk_start].to_pandas().values


@adapter_for('pyarrow.parquet.ParquetFile')
class PyArrowParquetFileAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)
        self._schema = data.schema
        # TODO: take pandas metadata index columns into account:
        # - display those columns as labels
        # - remove those columns from shape
        # - do not read those columns in get_values
        # pandas_metadata = data.schema.to_arrow_schema().pandas_metadata
        # index_columns = pandas_metadata['index_columns']
        meta = data.metadata
        num_rows_per_group = np.array([meta.row_group(i).num_rows
                                       for i in range(data.num_row_groups)])
        self._group_ends = num_rows_per_group.cumsum()
        self._cached_table = None
        self._cached_table_h_start = None
        self._cached_table_v_start = None

    def shape2d(self):
        meta = self.data.metadata
        return meta.num_rows, meta.num_columns

    def get_hlabels_values(self, start, stop):
        return [self._schema.names[start:stop]]

    # TODO: provide caching in a base class
    def _is_chunk_cached(self, h_start, v_start, h_stop, v_stop):
        cached_table = self._cached_table
        if cached_table is None:
            return False
        cached_v_start = self._cached_table_v_start
        cached_h_start = self._cached_table_h_start
        return (h_start >= cached_h_start and
                h_stop <= cached_h_start + cached_table.shape[1] and
                v_start >= cached_v_start and
                v_stop <= cached_v_start + cached_table.shape[0])

    def get_values(self, h_start, v_start, h_stop, v_stop):
        if self._is_chunk_cached(h_start, v_start, h_stop, v_stop):
            logger.debug("cache hit !")
            table = self._cached_table
            table_h_start = self._cached_table_h_start
            table_v_start = self._cached_table_v_start
        else:
            logger.debug("cache miss !")
            start_row_group, stop_row_group = (
                # - 1 because the last row is not included
                np.searchsorted(self._group_ends, [v_start, v_stop - 1],
                                side='right'))
            # - 1 because _group_ends stores row group ends and we want the start
            table_h_start = h_start
            table_v_start = (
                self._group_ends[start_row_group - 1] if start_row_group > 0 else 0)
            row_groups = range(start_row_group, stop_row_group + 1)
            column_names = self._schema.names[h_start:h_stop]
            f = self.data
            table = f.read_row_groups(row_groups, columns=column_names)
            self._cached_table = table
            self._cached_table_h_start = table_h_start
            self._cached_table_v_start = table_v_start

        chunk = table[v_start - table_v_start:v_stop - table_v_start]
        # not going via to_pandas() because it "eats" index columns
        columns = chunk.columns[h_start - table_h_start:h_stop - table_h_start]
        np_columns = [c.to_numpy() for c in columns]
        try:
            return np.stack(np_columns, axis=1)
        except np.exceptions.DTypePromotionError:
            return np.stack(np_columns, axis=1, dtype=object)


@adapter_for('pyarrow.Table')
class PyArrowTableAdapter(AbstractColumnarAdapter):
    def shape2d(self):
        # TODO: take pandas metadata index columns into account:
        # - display those columns as labels
        # - remove those columns from shape
        # - do not read those columns in get_values
        # self.data.schema.pandas_metadata
        return self.data.shape

    def get_hlabels_values(self, start, stop):
        return [self.data.column_names[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        chunk = self.data[v_start:v_stop].select(range(h_start, h_stop))
        # not going via to_pandas() because it "eats" index columns
        np_columns = [c.to_numpy() for c in chunk.columns]
        try:
            return np.stack(np_columns, axis=1)
        except np.exceptions.DTypePromotionError:
            return np.stack(np_columns, axis=1, dtype=object)


@adapter_for('polars.DataFrame')
@adapter_for('narwhals.DataFrame')
class PolarsDataFrameAdapter(AbstractColumnarAdapter):
    def shape2d(self):
        return self.data.shape

    def get_hlabels_values(self, start, stop):
        return [self.data.columns[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        # Going via Pandas instead of directly using to_numpy() because this
        # has a better behavior for datetime columns (e.g. pl_df3).
        # Otherwise, Polars converts datetimes to floats instead using a numpy
        # object array
        return self.data[v_start:v_stop, h_start:h_stop].to_pandas().values


@adapter_for('polars.LazyFrame')
class PolarsLazyFrameAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        import polars as pl
        assert isinstance(data, pl.LazyFrame)

        super().__init__(data=data, attributes=attributes)
        self._schema = data.collect_schema()
        self._columns = self._schema.names()
        # TODO: this is often slower than computing the "first window" data
        #       so we could try to use a temporary value and
        #       fill the real height as we go like for CSV files
        self._height = data.select(pl.len()).collect(engine='streaming').item()

    def shape2d(self):
        return self._height, len(self._schema)

    def get_hlabels_values(self, start, stop):
        return [self._columns[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        subset = self.data[v_start:v_stop].select(self._columns[h_start:h_stop])
        df = subset.collect(engine='streaming')
        # Going via Pandas instead of directly using to_numpy() because this
        # has a better behavior for datetime columns (e.g. pl_df3).
        # Otherwise, Polars converts datetimes to floats instead using a numpy
        # object array
        return df.to_pandas().values


@adapter_for('narwhals.LazyFrame')
class NarwhalsLazyFrameAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        import narwhals as nw
        assert isinstance(data, nw.LazyFrame)

        super().__init__(data=data, attributes=attributes)
        self._schema = data.collect_schema()
        self._columns = self._schema.names()
        # TODO: this is often slower than computing the "first window" data
        #       so we could try to use a temporary value and
        #       fill the real height as we go like for CSV files
        # TODO: engine='streaming' is not part of the narwhals API (it
        #       is forwarded to the underlying engine) so it will work with
        #       Polars but probably not other engines)
        self._height = data.select(nw.len()).collect(engine='streaming').item()
        self._wh_index = data.with_row_index('_index')

    def shape2d(self):
        return self._height, len(self._schema)

    def get_hlabels_values(self, start, stop):
        return [self._columns[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        nw = sys.modules['narwhals']
        # narwhals LazyFrame does not support slicing, so we have to
        # resort to this awful workaround which is MUCH slower than native
        # polars
        filter_ = (nw.col('_index') >= v_start) & (nw.col('_index') < v_stop)
        # .select also implicitly drops _index
        lazy_sub_df = self._wh_index.filter(filter_).select(self._columns[h_start:h_stop])
        return lazy_sub_df.collect(engine='streaming').to_numpy()


@adapter_for('iode.Variables')
class IodeVariablesAdapter(AbstractAdapter):
    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)
        self._periods = data.periods

    def shape2d(self):
        return len(self.data), len(self._periods)

    def get_hlabels_values(self, start, stop):
        return [[str(p) for p in self._periods[start:stop]]]

    def get_vlabels_values(self, start, stop):
        get_name = self.data.get_name
        return [[get_name(i)] for i in range(start, stop)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        get_name = self.data.get_name
        names = [get_name(i) for i in range(v_start, v_stop)]
        first_period = self._periods[h_start]
        # - 1 because h_stop itself is exlusive while iode stop period is inclusive
        last_period = self._periods[h_stop - 1]
        return self.data[names, first_period:last_period].to_numpy()


@adapter_for('iode.Table')
class IodeTableAdapter(AbstractAdapter):
    def shape2d(self):
        # TODO: ideally, width should be self.data.nb_columns, but we need
        #       to handle column spans in the model for that
        #       see: https://runebook.dev/en/articles/qt/qtableview/columnSpan
        return len(self.data), 1

    def get_hlabels_values(self, start, stop):
        return [['']]

    def get_vlabels_values(self, start, stop):
        return [[str(self.data[i].line_type)] for i in range(start, stop)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return [[str(self.data[i])] for i in range(v_start, v_stop)]


class AbstractIodeSimpleListAdapter(AbstractAdapter):
    def shape2d(self):
        return len(self.data), 1

    def get_hlabels_values(self, start, stop):
        return [[self._COLUMN_NAME]]

    def get_vlabels_values(self, start, stop):
        get_name = self.data.get_name
        return [[get_name(i)] for i in range(start, stop)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        indices_getter = self.data.i
        return [[str(indices_getter[i])]
                for i in range(v_start, v_stop)]


class AbstractIodeObjectListAdapter(AbstractAdapter):
    _ATTRIBUTES = []

    def shape2d(self):
        return len(self.data), len(self._ATTRIBUTES)

    def get_hlabels_values(self, start, stop):
        return [[attr.capitalize() for attr in self._ATTRIBUTES[start:stop]]]

    def get_vlabels_values(self, start, stop):
        get_name = self.data.get_name
        return [[get_name(i)] for i in range(start, stop)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        attrs = self._ATTRIBUTES[h_start:h_stop]
        indices_getter = self.data.i
        objects = [indices_getter[i] for i in range(v_start, v_stop)]
        return [[getattr(obj, attr) for attr in attrs]
                for obj in objects]


@adapter_for('iode.Comments')
class IodeCommentsAdapter(AbstractIodeSimpleListAdapter):
    _COLUMN_NAME = 'Comment'


@adapter_for('iode.Identities')
class IodeIdentitiesAdapter(AbstractIodeSimpleListAdapter):
    _COLUMN_NAME = 'Identity'


@adapter_for('iode.Lists')
class IodeListsAdapter(AbstractIodeSimpleListAdapter):
    _COLUMN_NAME = 'List'


@adapter_for('iode.Tables')
class IodeTablesAdapter(AbstractIodeObjectListAdapter):
    _ATTRIBUTES = ['title', 'language']

    def cell_activated(self, row_idx, column_idx):
        return self.data.i[row_idx]


@adapter_for('iode.Scalars')
class IodeScalarsAdapter(AbstractIodeObjectListAdapter):
    _ATTRIBUTES = ['value', 'std', 'relax']


@adapter_for('iode.Equations')
class IodeEquationsAdapter(AbstractAdapter):
    _COLNAMES = ['lec', 'method', 'sample', 'block',
                 'fstat', 'r2adj', 'dw', 'loglik',
                 'date']
    _SIMPLE_ATTRS = {'block', 'date', 'lec', 'method', 'sample'}

    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)

    def shape2d(self):
        return len(self.data), len(self._COLNAMES)

    def get_hlabels_values(self, start, stop):
        return [self._COLNAMES[start:stop]]

    def get_vlabels_values(self, start, stop):
        get_name = self.data.get_name
        return [[get_name(i)] for i in range(start, stop)]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        """*_stop are exclusive"""
        colnames = self._COLNAMES[h_start:h_stop]
        indices_getter = self.data.i
        simple_attrs = self._SIMPLE_ATTRS
        res = []
        for i in range(v_start, v_stop):
            try:
                eq = indices_getter[i]
                tests = eq.tests
                row = [str(getattr(eq, colname)).replace('\n', '')
                           if colname in simple_attrs else tests[colname]
                       for colname in colnames]
            except Exception:
                row = ['<bad value>'
                           if colname in simple_attrs else np.nan
                       for colname in colnames]
            res.append(row)
        return np.array(res, dtype=object)


@path_adapter_for('.av', 'iode')
@path_adapter_for('.var', 'iode')
class IodeVariablesPathAdapter(IodeVariablesAdapter):
    @classmethod
    def open(cls, fpath):
        iode = sys.modules['iode']
        iode.variables.load(str(fpath))
        return iode.variables


@path_adapter_for('.as', 'iode')
@path_adapter_for('.scl', 'iode')
class IodeScalarsPathAdapter(IodeScalarsAdapter):
    @classmethod
    def open(cls, fpath):
        iode = sys.modules['iode']
        iode.scalars.load(str(fpath))
        return iode.scalars


@path_adapter_for('.ac', 'iode')
@path_adapter_for('.cmt', 'iode')
class IodeCommentsPathAdapter(IodeCommentsAdapter):
    @classmethod
    def open(cls, fpath):
        iode = sys.modules['iode']
        iode.comments.load(str(fpath))
        return iode.comments


@path_adapter_for('.at', 'iode')
@path_adapter_for('.tbl', 'iode')
class IodeTablesPathAdapter(IodeTablesAdapter):
    @classmethod
    def open(cls, fpath):
        iode = sys.modules['iode']
        iode.tables.load(str(fpath))
        return iode.tables


@path_adapter_for('.ae', 'iode')
@path_adapter_for('.eqs', 'iode')
class IodeEquationsPathAdapter(IodeEquationsAdapter):
    @classmethod
    def open(cls, fpath):
        iode = sys.modules['iode']
        iode.equations.load(str(fpath))
        return iode.equations


@path_adapter_for('.ai', 'iode')
@path_adapter_for('.idt', 'iode')
class IodeIdentitiesPathAdapter(IodeIdentitiesAdapter):
    @classmethod
    def open(cls, fpath):
        iode = sys.modules['iode']
        iode.identities.load(str(fpath))
        return iode.identities


@path_adapter_for('.al', 'iode')
@path_adapter_for('.lst', 'iode')
class IodeListsPathAdapter(IodeListsAdapter):
    @classmethod
    def open(cls, fpath):
        iode = sys.modules['iode']
        iode.lists.load(str(fpath))
        return iode.lists


@adapter_for('ibis.Table')
class IbisTableAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)
        self._columns = data.columns
        self._height = data.count().execute()

    def shape2d(self):
        return self._height, len(self._columns)

    def get_hlabels_values(self, start, stop):
        return [self._columns[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        lazy_sub_df = self.data[v_start:v_stop].select(self._columns[h_start:h_stop])
        return lazy_sub_df.to_pandas().values


# TODO: reuse NumpyStructuredArrayAdapter
@adapter_for('tables.File')
class PyTablesFileAdapter(AbstractColumnarAdapter):
    _COLNAMES = ['Name']

    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)

    def shape2d(self):
        return self.data.root._v_nchildren, 1

    def get_hlabels_values(self, start, stop):
        return [self._COLNAMES[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        subnodes = self.data.list_nodes('/')
        return [[group._v_name][h_start:h_stop]
                for group in subnodes[v_start:v_stop]]

    def cell_activated(self, row_idx, column_idx):
        groups = self.data.list_nodes('/')
        return groups[row_idx]


class PyTablesGroupAdapter(AbstractColumnarAdapter):
    _COLNAMES = ['Name']

    def shape2d(self):
        return self.data._v_nchildren, 1

    def get_hlabels_values(self, start, stop):
        return [self._COLNAMES[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        subnodes = self.data._f_list_nodes()
        return [[group._v_name][h_start:h_stop]
                for group in subnodes[v_start:v_stop]]

    def cell_activated(self, row_idx, column_idx):
        subnodes = self.data._f_list_nodes()
        return subnodes[row_idx]


class PyTablesPandasFrameAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)
        attrs = data._v_attrs
        assert hasattr(attrs, 'nblocks') and attrs.nblocks == 1, "not implemented for nblocks > 1"
        assert hasattr(attrs, 'axis0_variety') and attrs.axis0_variety in {'regular', 'multi'}
        assert hasattr(attrs, 'axis1_variety') and attrs.axis1_variety in {'regular', 'multi'}
        self._axis0_variety = attrs.axis0_variety
        self._axis1_variety = attrs.axis1_variety
        self._encoding = getattr(attrs, 'encoding', None)

    def shape2d(self):
        return self.data.block0_values.shape

    def _get_axis_names(self, axis_num: int) -> list[str]:
        group = self.data
        attrs = group._v_attrs
        if getattr(attrs, f'axis{axis_num}_variety') == 'regular':
            axis_node = group._f_get_child(f'axis{axis_num}')
            return [axis_node._v_attrs.name]
        else:
            nlevels = getattr(attrs, f'axis{axis_num}_nlevels')
            return [
                group._f_get_child(f'axis{axis_num}_level{i}')._v_attrs.name
                for i in range(nlevels)
            ]

    def get_hnames(self):
        return self._get_axis_names(0)

    def get_vnames(self):
        return self._get_axis_names(1)

    def _get_axis_labels(self, axis_num: int, start: int, stop: int) -> np.ndarray:
        group = self.data
        attrs = group._v_attrs
        if getattr(attrs, f'axis{axis_num}_variety') == 'regular':
            axis_node = group._f_get_child(f'axis{axis_num}')
            labels = axis_node[start:stop].reshape(1, -1)
            kind = axis_node._v_attrs.kind
            if kind == 'string' and self._encoding is not None:
                labels = np.char.decode(labels, encoding=self._encoding)
        else:
            chunks = []
            has_strings = False
            has_non_strings = False
            nlevels = getattr(attrs, f'axis{axis_num}_nlevels')
            for i in range(nlevels):
                label_node = group._f_get_child(f'axis{axis_num}_label{i}')
                chunk_label_x = label_node[start:stop]
                max_label_x = chunk_label_x.max()
                level_node = group._f_get_child(f'axis{axis_num}_level{i}')
                axis_level_x = level_node[:max_label_x + 1]
                chunk_level_x = axis_level_x[chunk_label_x]
                kind = level_node._v_attrs.kind
                if kind == 'string':
                    has_strings = True
                    if self._encoding is not None:
                        chunk_level_x = np.char.decode(chunk_level_x, encoding=self._encoding)
                else:
                    has_non_strings = True
                chunks.append(chunk_level_x)
            if has_strings and has_non_strings:
                labels = np.stack(chunks, axis=0, dtype=object)
            else:
                labels = np.stack(chunks, axis=0)
        return labels

    def get_hlabels_values(self, start, stop):
        return self._get_axis_labels(0, start, stop)

    def get_vlabels_values(self, start, stop):
        return self._get_axis_labels(1, start, stop).transpose()

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return self.data.block0_values[v_start:v_stop, h_start:h_stop]


@adapter_for('tables.Group')
def dispatch_pytables_group_to_adapter(data):
    # distinguish between "normal" pytables Group and Pandas frames
    attrs = data._v_attrs
    if hasattr(attrs, 'pandas_type') and attrs.pandas_type == 'frame':
        return PyTablesPandasFrameAdapter
    else:
        return PyTablesGroupAdapter


@adapter_for('tables.Table')
class PyTablesTableAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)

    def shape2d(self):
        return len(self.data), len(self.data.dtype.names)

    def get_hlabels_values(self, start, stop):
        return [self._get_col_names(start, stop)]

    def _get_col_names(self, start, stop):
        return list(self.data.dtype.names[start:stop])

    def get_values(self, h_start, v_start, h_stop, v_stop):
        # TODO: when we scroll horizontally, we fetch the data over
        #       and over while we could only fetch it once
        #       given that pytables fetches entire rows anyway.
        #       Several solutions:
        #       * cache "current" rows in the adapter
        #       * have a way for the arraymodel to ask the adapter for the minimum buffer size
        #       * allow the adapter to return more data than what the model asked for and have the model actually
        #         use/take that extra data into account. This would require the adapter to return
        #         real_h_start, real_v_start (stop values can be deduced) in addition to actual values
        array = self.data[v_start:v_stop]
        return [tuple(row_data)[h_start:h_stop] for row_data in array]


@adapter_for('tables.Array')
class PyTablesArrayAdapter(NumpyHomogeneousArrayAdapter):
    def shape2d(self):
        if self.data.ndim == 1:
            return self.data.shape + (1,)
        else:
            return nd_shape_to_2d(self.data.shape)

    def get_vlabels_values(self, start, stop):
        shape = self.data.shape
        ndim = self.data.ndim
        if ndim == 1:
            shape += (1,)
        if ndim > 0:
            vlabels = Product([range(axis_len) for axis_len in shape[:-1]])
            return vlabels[start:stop]
        else:
            return [['']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        data = self.data
        if data.ndim == 1:
            return data[v_start:v_stop].reshape(-1, 1)
        elif data.ndim == 2:
            return data[v_start:v_stop, h_start:h_stop]
        else:
            raise NotImplementedError('>2d not implemented yet')


@path_adapter_for('.h5', 'tables')
class H5PathAdapter(PyTablesFileAdapter):
    @classmethod
    def open(cls, fpath):
        tables = sys.modules['tables']
        return tables.open_file(fpath)


# TODO: options to display as hex or decimal
# >>> s = f.read(20)
# >>> s
# b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc2\xea\x81\xb3\x14\x11\xcf\xbd
@adapter_for('_io.BufferedReader')
class BinaryFileAdapter(AbstractAdapter):
    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)
        self._nbytes = os.path.getsize(data.name)
        self._width = 16

    def shape2d(self):
        return math.ceil(self._nbytes / self._width), self._width

    def get_vlabels_values(self, start, stop):
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
            except Exception:
                buffer1d = np.append(buffer1d, np.zeros(filler_size, dtype='u1'))

        # change undisplayable characters to '.'
        buffer1d = np.where((buffer1d < 32) | (buffer1d >= 128),
                            ord('.'),
                            buffer1d).view('S1')

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


@adapter_for('_io.TextIOWrapper')
class TextFileAdapter(AbstractAdapter):
    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)
        # TODO: we should check at regular interval, this hasn't changed
        self._nbytes = os.path.getsize(self.data.name)
        self._lines_end_index = []
        self._fully_indexed = False
        self._encoding = None

        # sniff a small chunk so that we can compute an approximate number of lines
        # TODO: instead of opening and closing the file over and over, we
        #       should implement a mechanism to keep the file open while it is
        #       displayed and close it if another variable is selected.
        #       That might prevent the file from being deleted (by an external tool),
        #       which could be both annoying and practical.
        with self._binary_file as f:
            self._index_up_to(f, 1, chunk_size=64 * KB, max_time=0.05)

    @property
    def _binary_file(self):
        return open(self.data.name, 'rb')

    @property
    def _avg_bytes_per_line(self):
        lines_end_index = self._lines_end_index
        if lines_end_index:
            return lines_end_index[-1] / len(lines_end_index)
        elif self._nbytes:
            return self._nbytes
        else:
            return 1

    @property
    def _num_lines(self):
        """returns estimated number of lines"""
        if self._nbytes == 0:
            return 0
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
            logger.debug(f"trying to index up to {approx_v_stop}")
            start_time = time.perf_counter()
            chunk_start = self._lines_end_index[-1] if self._lines_end_index else 0
            f.seek(chunk_start)
            # TODO: check for off by one error with v_stop
            while (time.perf_counter() - start_time < max_time) and (len(self._lines_end_index) < approx_v_stop) and \
                    not self._fully_indexed:

                # TODO: if we are beyond v_start, we should store the chunks to avoid reading them twice from disk
                #       (once for indexing then again for getting the data)
                chunk = f.read(chunk_size)
                if self._encoding is None:
                    self._detect_encoding(chunk)

                line_end_char = b'\n'
                index_line_ends(chunk, self._lines_end_index, offset=chunk_start, c=line_end_char)
                length_read = len(chunk)
                # FIXME: this test is buggy.
                #        * if there was exactly chunk_size left to read, the file might never
                #          be marked as fully indexed
                #        * I think there are other (rare) reasons why a read can return
                #          less bytes than asked for
                if length_read < chunk_size:
                    self._fully_indexed = True
                    # add implicit line end at the end of the file if there isn't an explicit one
                    file_length = chunk_start + length_read
                    file_last_char_pos = file_length - len(line_end_char)
                    if not self._lines_end_index or self._lines_end_index[-1] != file_last_char_pos:
                        self._lines_end_index.append(file_length)
                chunk_start += length_read

    def _detect_encoding(self, chunk):
        try:
            import charset_normalizer
            chartset_match = charset_normalizer.from_bytes(chunk).best()
            self._encoding = chartset_match.encoding
            logger.debug(f"encoding detected as {self._encoding}")
        except ImportError:
            logger.debug("could not import 'charset_normalizer' => cannot detect encoding")

    def get_vlabels_values(self, start, stop):
        # we need to trigger indexing too (because get_vlabels happens before get_data) so that lines_indexed is correct
        # FIXME: get_data should not trigger indexing too if start/stop are the same
        with self._binary_file as f:
            self._index_up_to(f, stop)

        start, stop, step = slice(start, stop).indices(self._num_lines)
        lines_indexed = len(self._lines_end_index)
        return [[str(i) if i < lines_indexed else '~' + str(i)]
                for i in range(start, stop)]

    def _get_lines(self, start, stop):
        """stop is exclusive"""
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
                lines = self._decode_chunks_to_lines([chunk], stop - start)
                # lines = chunk.split(b'\n')
                # assert len(lines) == num_required_lines
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
                # first chunk size is what we *think* is necessary to get
                # num_lines_required
                chunk_size = stop_pos - start_pos
                # but then, if the number of lines we actually got (num_lines)
                # is not enough we will ask for more
                while num_lines < num_lines_required and stop_pos < max_stop_pos:
                    chunk = f.read(chunk_size)
                    chunks.append(chunk)
                    num_lines += chunk.count(b'\n')
                    stop_pos += len(chunk)
                    chunk_size = CHUNK_SIZE

                if approx_start:
                    # +1 and [1:] to remove first line so that we are sure the first line is complete
                    n_req_lines = num_lines_required + 1
                    lines = self._decode_chunks_to_lines(chunks, n_req_lines)[1:]
                else:
                    lines = self._decode_chunks_to_lines(chunks, num_lines_required)
                return lines

    def _decode_chunk(self, chunk: bytes):
        try:
            return chunk.decode(self._encoding)
        except UnicodeDecodeError:
            old_encoding = self._encoding
            # try to find another encoding
            self._detect_encoding(chunk)
            logger.debug(f"Could not decode chunk using {old_encoding}")
            logger.debug(f"Trying again using {self._encoding} and ignoring "
                         f"errors")
            return chunk.decode(self._encoding, errors='replace')

    def _decode_chunks_to_lines(self, chunks: list, num_required_lines: int):
        r"""
        Parameters
        ----------
        chunks : list
            List of chunks. str and bytes are both supported but should not be mixed (all chunks must
            have the same type than the first chunk).
        """
        if not chunks:
            return []

        # TODO: we could have more efficient code:
        #  * only decode as many chunks as necessary to get num_required_lines
        #  * only join as many chunks as necessary to get num_required_lines
        if self._encoding is not None:
            assert isinstance(chunks[0], bytes)
            chunks = [self._decode_chunk(chunk) for chunk in chunks]

        sep = b'' if isinstance(chunks[0], bytes) else ''
        lines = sep.join(chunks).splitlines()
        return lines[:num_required_lines]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        """*_stop are exclusive"""
        return [[line] for line in self._get_lines(v_start, v_stop)]


@path_adapter_for('.parquet', 'pyarrow.parquet')
class ParquetPathAdapter(PyArrowParquetFileAdapter):
    @classmethod
    def open(cls, fpath):
        pq = sys.modules['pyarrow.parquet']
        return pq.ParquetFile(fpath)


@path_adapter_for('.bat')
@path_adapter_for('.cfg')
@path_adapter_for('.md')
@path_adapter_for('.py')
@path_adapter_for('.rst')
@path_adapter_for('.sh')
@path_adapter_for('.toml')
@path_adapter_for('.txt')
@path_adapter_for('.yaml')
@path_adapter_for('.yml')
class TextPathAdapter(TextFileAdapter):
    @classmethod
    def open(cls, fpath):
        return open(fpath, 'rt')


@path_adapter_for('.xlsx', 'xlwings')
@path_adapter_for('.xls', 'xlwings')
class XlsxPathAdapter(WorkbookAdapter):
    @classmethod
    def open(cls, fpath):
        return la.open_excel(fpath)


class CsvFileAdapter(TextFileAdapter):
    def __init__(self, data, attributes):
        # we know the module is loaded but it is not in the current namespace
        csv = sys.modules['csv']
        TextFileAdapter.__init__(self, data=data, attributes=attributes)
        if self._nbytes > 0:
            first_line = self._get_lines(0, 1)
            assert len(first_line) == 1
            reader = csv.reader([first_line[0]])
            self._colnames = next(reader)
        else:
            self._colnames = []

    # for large files, this is approximate
    def shape2d(self):
        # - 1 for header row
        return self._num_lines - 1, len(self._colnames)

    def get_hlabels_values(self, start, stop):
        return [self._colnames[start:stop]]

    def get_vlabels_values(self, start, stop):
        # + 1 for header row
        return super().get_vlabels_values(start + 1, stop + 1)

    def get_values(self, h_start, v_start, h_stop, v_stop):
        """*_stop are exclusive"""
        # + 1 because the header row is not part of the data but _get_lines works
        # on the actual file lines
        lines = self._get_lines(v_start + 1, v_stop + 1)
        if not lines:
            return []
        # we know the module is loaded but it is not in the current namespace
        csv = sys.modules['csv']
        # Note that csv reader actually needs a line-based input
        reader = csv.reader(lines)
        return [line[h_start:h_stop] for line in reader]


@path_adapter_for('.csv', 'csv')
class CsvPathAdapter(CsvFileAdapter):
    @classmethod
    def open(cls, fpath):
        return open(fpath, 'rt')


# This is a Path adapter (it handles Path objects) because pyreadstat has no
# object representing open files
@path_adapter_for('.sas7bdat', 'pyreadstat')
class Sas7BdatPathAdapter(AbstractColumnarAdapter):
    # data must be a Path object
    def __init__(self, data, attributes=None):
        assert isinstance(data, Path)
        # we know the module is loaded but it is not in the current namespace
        pyreadstat = sys.modules['pyreadstat']
        super().__init__(data, attributes=attributes)
        empty_df, meta = pyreadstat.read_sas7bdat(data, metadataonly=True)
        self._colnames = meta.column_names
        self._numrows = meta.number_rows

    def shape2d(self):
        return self._numrows, len(self._colnames)

    def get_hlabels_values(self, start, stop):
        return [self._colnames[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        # we know the module is loaded but it is not in the current namespace
        pyreadstat = sys.modules['pyreadstat']
        used_cols = self._colnames[h_start:h_stop]
        df, meta = pyreadstat.read_sas7bdat(self.data, row_offset=v_start,
                                            row_limit=v_stop - v_start,
                                            usecols=used_cols)
        return df.values




@adapter_for('pstats.Stats')
class ProfilingStatsAdapter(AbstractColumnarAdapter):
    # we display everything except callers
    _COLNAMES = ['filepath', 'line num', 'func. name',
                 'ncalls (non rec)', 'ncalls (total)',
                 'tottime', 'cumtime']

    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)
        self._keys = list(data.stats.keys())

    def shape2d(self):
        return len(self._keys), len(self._COLNAMES)

    def get_hlabels_values(self, start, stop):
        return [self._COLNAMES[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        """*_stop are exclusive"""

        func_calls = self._keys[v_start:v_stop]
        stats = self.data.stats
        call_details = [stats[k] for k in func_calls]
        # we display everything except callers
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


@adapter_for(SQLiteTable)
class SQLiteTableAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        assert isinstance(data, SQLiteTable)
        super().__init__(data=data, attributes=attributes)
        table_name = self.data.name
        cur = self.data.con.cursor()
        cur.execute(f"SELECT count(*) FROM {table_name}")
        self._numrows = cur.fetchone()[0]
        cur.execute(f"SELECT * FROM {table_name} LIMIT 1")
        self._columns = [col_descr[0] for col_descr in cur.description]
        cur.close()

    def shape2d(self):
        return self._numrows, len(self._columns)

    def get_hlabels_values(self, start, stop):
        return [self._columns[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        cur = self.data.con.cursor()
        cols = self._columns[h_start:h_stop]
        cur.execute(f"SELECT {', '.join(cols)} FROM {self.data.name} "
                    f"LIMIT {v_stop - v_start} OFFSET {v_start}")
        rows = cur.fetchall()
        cur.close()
        return rows


@adapter_for('sqlite3.Connection')
class SQLiteConnectionAdapter(AbstractAdapter):
    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)
        # as of python3.12, sqlite3.Cursor is not context manager friendly
        cur = data.cursor()
        cur.execute(SQLITE_LIST_TABLES_QUERY)
        self._table_names = [row[0] for row in cur.fetchall()]
        cur.close()

    def shape2d(self):
        return len(self._table_names), 1

    def get_hlabels_values(self, start, stop):
        return [['Name']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return [[name] for name in self._table_names[v_start:v_stop]]

    def cell_activated(self, row_idx, column_idx):
        table_name = self._table_names[row_idx]
        return SQLiteTable(self.data, table_name)


DUCKDB_LIST_TABLES_QUERY = "SHOW TABLES"

@adapter_for('duckdb.DuckDBPyRelation')
class DuckDBRelationAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)
        self._numrows = len(data)
        self._columns = data.columns

    def shape2d(self):
        return self._numrows, len(self._columns)

    def get_hlabels_values(self, start, stop):
        return [self._columns[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        cols = self._columns[h_start:h_stop]
        num_rows = v_stop - v_start
        rows = self.data.limit(num_rows, offset=v_start)
        subset = rows.select(*cols)
        return subset.fetchall()


@adapter_for('duckdb.DuckDBPyConnection')
class DuckDBConnectionAdapter(AbstractAdapter):
    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)
        self._table_names = [
            row[0] for row in data.sql(DUCKDB_LIST_TABLES_QUERY).fetchall()
        ]

    def shape2d(self):
        return len(self._table_names), 1

    def get_hlabels_values(self, start, stop):
        return [['Table Name']]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return [[name] for name in self._table_names[v_start:v_stop]]

    def cell_activated(self, row_idx, column_idx):
        table_name = self._table_names[row_idx]
        return self.data.table(table_name)


@path_adapter_for('.ddb', 'duckdb')
@path_adapter_for('.duckdb', 'duckdb')
class DuckDBPathAdapter(DuckDBConnectionAdapter):
    @classmethod
    def open(cls, fpath):
        duckdb = sys.modules['duckdb']
        return duckdb.connect(fpath)


@adapter_for('zipfile.ZipFile')
class ZipFileAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)

        infolist = data.infolist()
        infolist.sort(key=lambda info: (not info.is_dir(), info.filename))
        self._infolist = infolist
        self._list = [(info.filename,
                       datetime(*info.date_time).strftime('%d/%m/%Y %H:%M'),
                       '<directory>' if info.is_dir() else info.file_size)
                      for info in infolist]
        self._colnames = ['Name', 'Time Modified', 'Size']

    def shape2d(self):
        return len(self._list), len(self._colnames)

    def get_hlabels_values(self, start, stop):
        return [self._colnames[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return [row[h_start:h_stop]
                for row in self._list[v_start:v_stop]]

    def cell_activated(self, row_idx, column_idx):
        import zipfile
        info = self._infolist[row_idx]
        if info.is_dir():
            return zipfile.Path(self.data, info.filename)
        else:
            # do nothing for now
            return None
            # TODO: this returns a zipfile.ZipExtFile which is a file-like
            #       object but it does not inherit from io.BufferedReader so no
            #       adapter corresponds. We should add an adapter for
            #       zipfile.ZipExtFile
            # return self.data.open(info.filename)


@path_adapter_for('.zip', 'zipfile')
class ZipPathAdapter(ZipFileAdapter):
    @classmethod
    def open(cls, fpath):
        zipfile = sys.modules['zipfile']
        return zipfile.ZipFile(fpath)


@adapter_for('zipfile.Path')
class ZipfilePathAdapter(AbstractColumnarAdapter):
    def __init__(self, data, attributes):
        super().__init__(data=data, attributes=attributes)
        zpath_objs = list(data.iterdir())
        zpath_objs.sort(key=lambda p: (not p.is_dir(), p.name))
        self._zpath_objs = zpath_objs
        self._list = [(p.name, '<DIR>' if p.is_dir() else '')
                      for p in zpath_objs]
        self._colnames = ['Name', 'Type']

    def shape2d(self):
        return len(self._list), len(self._colnames)

    def get_hlabels_values(self, start, stop):
        return [self._colnames[start:stop]]

    def get_values(self, h_start, v_start, h_stop, v_stop):
        return [row[h_start:h_stop]
                for row in self._list[v_start:v_stop]]

    def cell_activated(self, row_idx, column_idx):
        child_path = self._zpath_objs[row_idx]
        if child_path.is_dir():
            return child_path
        else:
            # for now, do nothing
            return None
