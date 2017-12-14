from __future__ import absolute_import, division, print_function

import numpy as np
import larray as la

from larray_editor.utils import Product, _LazyDimLabels, Axis, get_sample


class AbstractAdapter(object):
    def __init__(self, axes_model, hlabels_model, vlabels_model, data_model):
        self.current_filter = {}
        self.changes = {}
        self.original_data = None
        self.bg_value = None
        self.filtered_data = None

        self.set_models(axes_model=axes_model, hlabels_model=hlabels_model, vlabels_model=vlabels_model,
                        data_model=data_model)

    def set_models(self, axes_model, hlabels_model, vlabels_model, data_model):
        """Set models"""
        self.axes_model = axes_model
        self.hlabels_model = hlabels_model
        self.vlabels_model = vlabels_model
        self.data_model = data_model
        self.models = \
            {
            'axes': self.axes_model,
            'hlabels': self.hlabels_model,
            'vlabels': self.vlabels_model,
            'data': self.data_model
            }

    def get_data(self):
        """Return original data"""
        return self.original_data

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

    def get_internal_data(self, data):
        """Return internal data as a ND Numpy array"""
        raise NotImplementedError()

    def get_bg_value(self, bg_value):
        """Return bg_value as ND Numpy array or None.
        It must have the same shape as data if not None.
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

    def apply_changes(self, data, changes):
        """Apply changes to the original data"""
        raise NotImplementedError()

    def get_ndim(self, data):
        raise NotImplementedError()

    def get_size(self, data):
        raise NotImplementedError()

    def get_dtype(self, data):
        raise NotImplementedError()

    @property
    def ndim(self):
        return self.get_ndim(self.original_data)

    @property
    def size(self):
        return self.get_size(self.original_data)

    @property
    def dtype(self):
        return self.get_dtype(self.original_data)

    def _get_axes(self):
        return self.get_axes(self.filtered_data)

    @property
    def bgcolor_possible(self):
        return self.data_model.bgcolor_possible

    def columnCount(self, model):
        """Return number of columns.

        Parameters
        ----------
        model: str
            Model's name. Must be either 'axes' or 'hlabels' or 'vlabels' or 'data'.
        """
        return self.models[model].columnCount()

    def rowCount(self, model):
        """Return number of rows.

        Parameters
        ----------
        model: str
            Model's name. Must be either 'axes' or 'hlabels' or 'vlabels' or 'data'.
        """
        return self.models[model].rowCount()

    def _get_sample(self):
        """Return a sample of the internal data"""
        data = self.get_internal_data(self.filtered_data)
        # this will yield a data sample of max 200
        sample = get_sample(data, 200)
        return sample[np.isfinite(sample)]

    def set_format(self, digits, scientific, reset=True):
        """Set format.

        Parameters
        ----------
        digits : int
            Number of digits to display.
        scientific : boolean
            Whether or not to display values in scientific format.
        reset: boolean, optional
            Whether or not to reset the data model. Defaults to True.
        """
        type = self.dtype.type
        if type in (np.str, np.str_, np.bool_, np.bool, np.object_):
            fmt = '%s'
        else:
            format_letter = 'e' if scientific else 'f'
            fmt = '%%.%d%s' % (digits, format_letter)
        self.data_model.set_format(fmt, reset)

    def _paste_data(self, row_min, col_min, row_max, col_max, new_data):
        """Paste new data in Data model.

        Parameters
        ----------
        row_min : int
            Lower row where to paste new data.
        row_max : int
            Upper row where to paste new data.
        col_min : int
            Lower column where to paste new data.
        col_max : int
            Upper column where to paste new data.
        new_data : Numpy 2D array
            Data to be pasted.

        Returns
        -------
        tuple of QModelIndex or None
            Actual bounds (end bound is inclusive) if update was successful, None otherwise
        """
        return self.data_model.set_values(row_min, col_min, row_max, col_max, new_data)

    def _get_axes_names(self):
        return [axis.name for axis in self._get_axes()]

    def _set_axes_names(self, reset=True):
        axes_names = self._get_axes_names()
        if len(axes_names) >= 2:
            axes_names = axes_names[:-2] + [axes_names[-2] + '\\' + axes_names[-1]]
        axes_names = [[axis_name] for axis_name in axes_names] if len(axes_names) > 0 else [[]]
        self.axes_model.set_data(axes_names, reset)

    def _set_vlabels(self, reset=True):
        axes = self.get_axes(self.filtered_data)
        if len(axes) == 0:
            vlabels = [[]]
        elif len(axes) == 1:
            vlabels = [['']]
        else:
            vlabels = [axis.labels for axis in axes[:-1]]
            prod = Product(vlabels)
            vlabels = [_LazyDimLabels(prod, i) for i in range(len(vlabels))]
        self.vlabels_model.set_data(vlabels, reset)

    def _set_hlabels(self, reset=True):
        axes = self.get_axes(self.filtered_data)
        if len(axes) == 0:
            hlabels = [[]]
        else:
            hlabels = axes[-1].labels
            hlabels = Product([hlabels])
        self.hlabels_model.set_data(hlabels, reset)

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

    def _set_data(self, reset=True):
        """Feed the Data model with new data and update bg value if required"""
        # get filtered data as Numpy ND array
        np_data = self.get_internal_data(self.filtered_data)
        assert isinstance(np_data, np.ndarray)
        data_shape = np_data.shape
        # compute equivalent 2D shape
        shape_2D = self._get_shape_2D(np_data)
        assert shape_2D[0] * shape_2D[1] == np_data.size
        # feed the Data model with data reshaped as 2D array
        # use flag reset=False to avoid calling reset() several times
        self.data_model.set_data(np_data.reshape(shape_2D), reset)

    def _set_bg_value(self, reset=True):
        """Set the bg value in Data model"""
        # get filtered bg value as Numpy ND array or None
        np_bg_value = self.get_bg_value(self.filter_data(self.bg_value, self.current_filter))
        if np_bg_value is not None:
            data_shape = self.data_model.get_data().shape
            assert isinstance(np_bg_value, np.ndarray) and np_bg_value.size == np.prod(data_shape)
            np_bg_value = np_bg_value.reshape(data_shape)
        # set bg_value in Data model (bg_value reshaped as 2D array)
        # use flag reset=False to avoid calling reset() several times
        self.data_model.set_bg_value(np_bg_value, reset)

    def _set_bg_gradient(self, gradient):
        self.data_model.set_bg_gradient(gradient)

    def _set_changes(self):
        """Map all changes applied to raw data to equivalent changes to data holded by models"""
        # we cannot apply the changes directly to data because it might be a view
        assert isinstance(self.changes, dict)
        changes_2D = {}
        for key, value in self.changes.items():
            local_key = self._map_global_to_filtered(self.original_data, self.filtered_data, self.current_filter, key)
            if local_key is not None:
                changes_2D[local_key] = value
        self.data_model.set_changes(changes_2D)

    def _reset_minmax(self):
        self.data_model.reset_minmax()
        self.data_model.reset()

    def _update_models(self, reset_model):
        self._set_axes_names()
        self._set_hlabels()
        self._set_vlabels()
        self._set_data(reset=False)
        self._set_bg_value(reset=False)
        self._set_changes()
        if reset_model:
            self.data_model.reset()

    def update_filtered_data(self, reset_model=True):
        self.filtered_data = self.filter_data(self.original_data, self.current_filter)
        self._update_models(reset_model)

    def set_data(self, data, changes=None, bg_value=None):
        self.current_filter = {}
        self.original_data = self.prepare_data(data)
        self.bg_value = self.prepare_bg_value(bg_value)
        self.changes = {} if changes is None else changes
        self.update_filtered_data(reset_model=False)
        self._reset_minmax()

    def _change_filter(self, axis, indices):
        # must be done before changing self.current_filter
        self.update_changes()
        self.change_filter(self.original_data, self.current_filter, axis, indices)
        self.update_filtered_data()

    def update_changes(self):
        changes_2D = self.data_model.changes
        for key, value in changes_2D.items():
            self.changes[self._map_filtered_to_global(
                self.filtered_data, self.original_data, self.current_filter, key)] = value

    def clear_changes(self):
        self.changes.clear()
        self.data_model.changes.clear()

    def accept_changes(self):
        """Accept changes"""
        # update changes
        self.update_changes()
        # update internal data
        self.apply_changes(self.original_data, self.changes)
        # update models
        self.update_filtered_data()
        # clear changes
        self.clear_changes()
        # return modified data
        return self.original_data

    def reject_changes(self):
        """Reject changes"""
        # clear changes
        self.clear_changes()
        self._reset_minmax()


class LArrayDataAdapter(AbstractAdapter):
    def __init__(self, axes_model, hlabels_model, vlabels_model, data_model):
        AbstractAdapter.__init__(self, axes_model=axes_model, hlabels_model=hlabels_model, vlabels_model=vlabels_model,
                                 data_model=data_model)

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

    def get_internal_data(self, data):
        assert isinstance(data, la.LArray)
        return data.data

    def get_bg_value(self, bg_value):
        if bg_value is not None:
            assert isinstance(bg_value, la.LArray)
            return bg_value.data
        else:
            return bg_value

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