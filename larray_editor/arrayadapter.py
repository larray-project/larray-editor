from __future__ import absolute_import, division, print_function

import numpy as np
import larray as la

from larray_editor.utils import Product, _LazyDimLabels, Axis, get_sample


REGISTERED_ADAPTERS = {}

def register_adapter(type):
    """Class decorator to register new adapter

    Parameters
    ----------
    type : type
        Type associated with adapter class.
    """
    def decorate_class(cls):
        if type not in REGISTERED_ADAPTERS:
            REGISTERED_ADAPTERS[type] = cls
        return  cls
    return decorate_class


def get_adapter(data, changes, bg_value):
    if data is None:
        return None
    data_type = type(data)
    if data_type not in REGISTERED_ADAPTERS:
        raise TypeError("No Adapter implemented for data with type {}".format(data_type))
    adapter_cls = REGISTERED_ADAPTERS[data_type]
    return adapter_cls(data, changes, bg_value)


class AbstractAdapter(object):
    def __init__(self, data, changes, bg_value):
        self.data = data
        self.bg_value = bg_value
        self.changes = changes
        self.current_filter = {}
        self.update_filtered_data()

    # ===================== #
    #      PROPERTIES       #
    # ===================== #

    @property
    def data(self):
        return self._original_data

    @data.setter
    def data(self, original_data):
        assert original_data is not None, "{} does not accept None as input data".format(self.__class__)
        self._original_data = self.prepare_data(original_data)

    @property
    def bg_value(self):
        return self._bg_value

    @bg_value.setter
    def bg_value(self, bg_value):
        self._bg_value = self.prepare_bg_value(bg_value)

    @property
    def changes(self):
        return self._changes

    @changes.setter
    def changes(self, changes):
        if changes is None:
            self._changes = {}
        else:
            assert isinstance(changes, dict), "{} only accept None or a dict as input changes".format(self.__class__)
            self._changes = changes

    @property
    def ndim(self):
        return self.get_ndim(self._original_data)

    @property
    def size(self):
        return self.get_size(self._original_data)

    @property
    def dtype(self):
        return self.get_dtype(self._original_data)

    # ===================== #
    #  METHODS TO OVERRIDE  #
    # ===================== #

    def get_ndim(self, data):
        raise NotImplementedError()

    def get_size(self, data):
        raise NotImplementedError()

    def get_dtype(self, data):
        raise NotImplementedError()

    def prepare_data(self, data):
        """Must be overridden if data passed to set_data need some checks and/or transformations"""
        return data

    def prepare_bg_value(self, bg_value):
        """Must be overridden if bg_value passed to set_data need some checks and/or transformations"""
        return bg_value

    def filter_data(self, data, filter):
        """Return filtered data"""
        raise NotImplementedError()

    def get_axes(self, data):
        """Return list of :py:class:`Axis` or an empty list in case of a scalar or an empty array.
        """
        raise NotImplementedError()

    def _get_raw_data(self, data):
        """Return internal data as a ND Numpy array"""
        raise NotImplementedError()

    def _get_bg_value(self, bg_value):
        """Return bg_value as ND Numpy array or None.
        It must have the same shape as data if not None.
        """
        raise NotImplementedError()

    def from_selection(self, raw_data, axes_names, vlabels, hlabels):
        """Create and return an object of type managed by the adapter subclass.

        Parameters
        ----------
        raw_data : Numpy.ndarray
            Array of selected data.
        axes_names : list of string
            List of axis names
        vlabels : nested list
            Selected vertical labels
        hlabels: list
            Selected horizontal labels

        Returns
        -------
        Object of the type managed by the adapter subclass.
        """
        raise NotImplementedError()

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

    def _map_global_to_filtered(self, data, filtered_data, filter, key):
        """
        map global (unfiltered) ND key to local (filtered) 2D key

        Parameters
        ----------
        data : array
            Input array.
        filtered_data : array
            Filtered data.
        filter : dict
            Current filter.
        key: tuple
            Labels associated with the modified element of the non-filtered array.

        Returns
        -------
        tuple
            Positional index (row, column) of the modified data cell.
        """
        raise NotImplementedError()

    def _map_filtered_to_global(self, filtered_data, data, filter, key):
        """
        map local (filtered data) 2D key to global (unfiltered) ND key.

        Parameters
        ----------
        filtered_data : array
            Filtered data.
        data : array
            Input array.
        filter : dict
            Current filter.
        key: tuple
            Positional index (row, column) of the modified data cell.

        Returns
        -------
        tuple
            Labels associated with the modified element of the non-filtered array.
        """
        raise NotImplementedError()

    def change_filter(self, data, filter, axis, indices):
        """Update current filter for a given axis if labels selection from the array widget has changed

        Parameters
        ----------
        data : array
            Input array.
        filter: dict
            Dictionary {axis_id: labels} representing the current selection.
        axis: axis
             Axis for which selection has changed.
        indices: list of int
            Indices of selected labels.
        """
        raise NotImplementedError()

    def to_excel(self, data):
        """Export data to an Excel Sheet

        Parameters
        ----------
        data : array
            data to export.
        """
        raise NotImplementedError()

    def plot(self, data):
        """Return a matplotlib.Figure object using input data.

        Parameters
        ----------
        data : array
            Data to plot.

        Returns
        -------
        A matplotlib.Figure object.
        """
        raise NotImplementedError

    def apply_changes(self, data, changes):
        """Apply changes to the original data"""
        raise NotImplementedError()

    # =========================== #
    #       OTHER METHODS         #
    # =========================== #

    def get_axes_filtered_data(self):
        return self.get_axes(self.filtered_data)

    def get_sample(self):
        """Return a sample of the internal data"""
        data = self._get_raw_data(self.filtered_data)
        # this will yield a data sample of max 200
        sample = get_sample(data, 200)
        return sample[np.isfinite(sample)]

    def get_axes_names(self, fold_last_axis=False):
        axes_names = [axis.name for axis in self.get_axes_filtered_data()]
        if fold_last_axis and len(axes_names) >= 2:
            axes_names = axes_names[:-2] + [axes_names[-2] + '\\' + axes_names[-1]]
        axes_names = [[axis_name] for axis_name in axes_names] if len(axes_names) > 0 else [[]]
        return axes_names

    def get_vlabels(self):
        axes = self.get_axes(self.filtered_data)
        if len(axes) == 0:
            vlabels = [[]]
        elif len(axes) == 1:
            vlabels = [['']]
        else:
            vlabels = [axis.labels for axis in axes[:-1]]
            prod = Product(vlabels)
            vlabels = [_LazyDimLabels(prod, i) for i in range(len(vlabels))]
        return vlabels

    def get_hlabels(self):
        axes = self.get_axes(self.filtered_data)
        if len(axes) == 0:
            hlabels = [[]]
        else:
            hlabels = axes[-1].labels
            hlabels = Product([hlabels])
        return hlabels

    def _get_shape_2D(self, np_data):
        shape, ndim = np_data.shape, np_data.ndim
        if ndim == 0:
            shape_2D = (1, 1)
        elif ndim == 1:
            shape_2D = (1,) + shape
        elif ndim == 2:
            shape_2D = shape
        else:
            shape_2D = (np.prod(shape[:-1]), shape[-1])
        return shape_2D

    def get_raw_data(self):
        # get filtered data as Numpy ND array
        np_data = self._get_raw_data(self.filtered_data)
        assert isinstance(np_data, np.ndarray)
        # compute equivalent 2D shape
        shape_2D = self._get_shape_2D(np_data)
        assert shape_2D[0] * shape_2D[1] == np_data.size
        # return data reshaped as 2D array
        return np_data.reshape(shape_2D)

    def get_bg_value(self):
        # get filtered bg value as Numpy ND array or None
        if self.bg_value is None:
            return self.bg_value
        np_bg_value = self._get_bg_value(self.filter_data(self.bg_value, self.current_filter))
        # compute equivalent 2D shape
        shape_2D = self._get_shape_2D(np_bg_value)
        assert shape_2D[0] * shape_2D[1] == np_bg_value.size
        # return bg_value reshaped as 2D array if not None
        return np_bg_value.reshape(shape_2D)

    def get_model_changes(self):
        # we cannot apply the changes directly to data because it might be a view
        changes_2D = {}
        for key, value in self.changes.items():
            local_key = self._map_global_to_filtered(self.data, self.filtered_data, self.current_filter, key)
            if local_key is not None:
                changes_2D[local_key] = value
        return changes_2D

    def update_filtered_data(self):
        self.filtered_data = self.filter_data(self.data, self.current_filter)

    def _update_filter(self, axis, indices, data_model_changes):
        # must be done before to call update_filter method of data_adapter
        self.update_changes(data_model_changes)
        self.change_filter(self.data, self.current_filter, axis, indices)
        self.update_filtered_data()

    def update_changes(self, data_model_changes):
        for key, value in data_model_changes.items():
            self.changes[self._map_filtered_to_global(
                self.filtered_data, self.data, self.current_filter, key)] = value

    def clear_changes(self):
        self.changes.clear()

    def accept_changes(self, data_model_changes):
        """Accept changes"""
        # update internal changes
        self.update_changes(data_model_changes)
        # update internal data
        self.apply_changes(self.data, self.changes)


@register_adapter(np.ndarray)
@register_adapter(la.LArray)
class LArrayDataAdapter(AbstractAdapter):
    def __init__(self, data, changes, bg_value):
        AbstractAdapter.__init__(self, data=data, changes=changes, bg_value=bg_value)

    def get_ndim(self, data):
        return data.ndim

    def get_size(self, data):
        return data.size

    def get_dtype(self, data):
        return data.dtype

    def prepare_data(self, data):
        return la.aslarray(data)

    def prepare_bg_value(self, bg_value):
        return la.aslarray(bg_value) if bg_value is not None else None

    def filter_data(self, data, filter):
        if data is None:
            return data
        assert isinstance(data, la.LArray)
        if filter is None:
            return data
        else:
            assert isinstance(filter, dict)
            data = data[filter]
            return la.aslarray(data) if np.isscalar(data) else data

    def get_axes(self, data):
        assert isinstance(data, la.LArray)
        axes = data.axes
        # test data.size == 0 is required in case an instance built as LArray([]) is passed
        # test len(axes) == 0 is required when a user filters until to get a scalar
        if data.size == 0 or len(axes) == 0:
            return []
        else:
            return [Axis(axes.axis_id(axis), name, axis.labels) for axis, name in zip(axes, axes.display_names)]

    def _get_raw_data(self, data):
        assert isinstance(data, la.LArray)
        return data.data

    def _get_bg_value(self, bg_value):
        if bg_value is not None:
            assert isinstance(bg_value, la.LArray)
            return bg_value.data
        else:
            return bg_value

    # TODO: We may want to update this method the day LArray objects will also handle MultiIndex-like axes.
    def from_selection(self, raw_data, axes_names, vlabels, hlabels):
        axes = []
        # combine the N-1 first axes
        if len(axes_names) > 1:
            combined_axes_names = '_'.join(axes_names[:-1])
            combined_labels = ['_'.join([str(vlabels[i][j]) for i in range(len(vlabels))])
                               for j in range(len(vlabels[0]))]
            axes = [la.Axis(combined_labels, combined_axes_names)]
        # last axis
        axes += [la.Axis(hlabels, axes_names[-1])]
        return la.LArray(raw_data, axes)

    def move_axis(self, data, bg_value, old_index, new_index):
        assert isinstance(data, la.LArray)
        new_axes = data.axes.copy()
        new_axes.insert(new_index, new_axes.pop(new_axes[old_index]))
        data = data.transpose(new_axes)
        if bg_value is not None:
            assert isinstance(bg_value, la.LArray)
            bg_value = bg_value.transpose(new_axes)
        return data, bg_value

    def _map_filtered_to_global(self, filtered_data, data, filter, key):
        # transform local positional index key to (axis_ids: label) dictionary key.
        # Contains only displayed axes
        row, col = key
        labels = [filtered_data.axes[-1].labels[col]]
        for axis in reversed(filtered_data.axes[:-1]):
            row, position = divmod(row, len(axis))
            labels = [axis.labels[position]] + labels
        axes_ids = list(filtered_data.axes.ids)
        dkey = dict(zip(axes_ids, labels))
        # add the "scalar" parts of the filter to it (ie the parts of the
        # filter which removed dimensions)
        dkey.update({k: v for k, v in filter.items() if np.isscalar(v)})
        # re-transform it to tuple (to make it hashable/to store it in .changes)
        return tuple(dkey[axis_id] for axis_id in data.axes.ids)

    def _map_global_to_filtered(self, data, filtered_data, filter, key):
        assert isinstance(key, tuple) and len(key) == data.ndim
        dkey = {axis_id: axis_key for axis_key, axis_id in zip(key, data.axes.ids)}
        # transform global dictionary key to "local" (filtered) key by removing
        # the parts of the key which are redundant with the filter
        for axis_id, axis_filter in filter.items():
            axis_key = dkey[axis_id]
            if np.isscalar(axis_filter) and axis_key == axis_filter:
                del dkey[axis_id]
            elif not np.isscalar(axis_filter) and axis_key in axis_filter:
                pass
            else:
                # that key is invalid for/outside the current filter
                return None
        # transform (axis:label) dict key to positional ND key
        try:
            index_key = filtered_data._translated_key(dkey)
        except ValueError:
            return None
        # transform positional ND key to positional 2D key
        strides = np.append(1, np.cumprod(filtered_data.shape[1:-1][::-1], dtype=int))[::-1]
        return (index_key[:-1] * strides).sum(), index_key[-1]

    def change_filter(self, data, filter, axis, indices):
        axis_id = axis.id
        if not indices or len(indices) == len(axis):
            if axis_id in filter:
                del filter[axis_id]
        else:
            if len(indices) == 1:
                filter[axis_id] = axis.labels[indices[0]]
            else:
                filter[axis_id] = axis.labels[indices]

    def apply_changes(self, data, changes):
        axes = data.axes
        for k, v in changes.items():
            data.i[axes.translate_full_key(k)] = v
