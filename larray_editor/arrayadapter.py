from __future__ import absolute_import, division, print_function

import numpy as np
import larray as la

from larray_editor.utils import Product, _LazyDimLabels


class LArrayDataAdapter(object):
    def __init__(self, axes_model, hlabels_model, vlabels_model, data_model):
        # set models
        self.axes_model = axes_model
        self.hlabels_model = hlabels_model
        self.vlabels_model = vlabels_model
        self.data_model = data_model

        self.current_filter = {}
        self.changes = {}
        self.la_data = None
        self.bg_value = None
        self.filtered_data = None

    def get_axes_names(self):
        return self.filtered_data.axes.display_names

    def get_axes(self):
        axes = self.filtered_data.axes
        # test self.filtered_data.size == 0 is required in case an instance built as LArray([]) is passed
        # test len(axes) == 0 is required when a user filters until to get a scalar
        if self.filtered_data.size == 0 or len(axes) == 0:
            return [[]]
        else:
            axes_names = axes.display_names
            if len(axes_names) >= 2:
                axes_names = axes_names[:-2] + [axes_names[-2] + '\\' + axes_names[-1]]
            return [[axis_name] for axis_name in axes_names]

    def get_hlabels(self):
        axes = self.filtered_data.axes
        if self.filtered_data.size == 0 or len(axes) == 0:
            return [[]]
        else:
            # this is a lazy equivalent of:
            # return [(label,) for label in axes.labels[-1]]
            return Product([axes.labels[-1]])

    def get_vlabels(self):
        axes = self.filtered_data.axes
        if self.filtered_data.size == 0 or len(axes) == 0:
            return [[]]
        elif len(axes) == 1:
            return [['']]
        else:
            labels = axes.labels[:-1]
            prod = Product(labels)
            return [_LazyDimLabels(prod, i) for i in range(len(labels))]

    def get_2D_data(self):
        """Returns Numpy 2D ndarray"""
        data_ND = self.filtered_data.data
        shape_ND, ndim = data_ND.shape, data_ND.ndim
        if ndim == 0:
            shape_2D = (1, 1)
        elif ndim == 1:
            shape_2D = (1,) + shape_ND
        elif ndim == 2:
            shape_2D = shape_ND
        else:
            shape_2D = (np.prod(shape_ND[:-1]), shape_ND[-1])
        return data_ND.reshape(shape_2D)

    def get_changes_2D(self):
        # we cannot apply the changes directly to data because it might be a view
        changes_2D = {}
        for k, v in self.changes.items():
            local_key = self._map_global_to_filtered(k)
            if local_key is not None:
                changes_2D[local_key] = v
        return changes_2D

    def get_bg_value_2D(self, shape_2D):
        if self.bg_value is not None:
            filtered_bg_value = self.bg_value[self.current_filter]
            if np.isscalar(filtered_bg_value):
                filtered_bg_value = la.aslarray(filtered_bg_value)
            return filtered_bg_value.data.reshape(shape_2D)
        else:
            return None

    # XXX: or create two methods?:
    # - set_data (which reset the current filter)
    # - update_data (which sets new data but keeps current filter unchanged)
    def set_data(self, data, bg_value=None):
        assert isinstance(data, la.LArray)
        self.current_filter = {}
        self.changes = {}
        self.la_data = la.aslarray(data)
        self.bg_value = la.aslarray(bg_value) if bg_value is not None else None
        self.update_filtered_data()
        self.data_model.reset_minmax()
        self.data_model.reset()

    def update_filtered_data(self):
        assert isinstance(self.la_data, la.LArray)
        self.filtered_data = self.la_data[self.current_filter]

        if np.isscalar(self.filtered_data):
            self.filtered_data = la.aslarray(self.filtered_data)

        axes = self.get_axes()
        hlabels = self.get_hlabels()
        vlabels = self.get_vlabels()
        data_2D = self.get_2D_data()
        changes_2D = self.get_changes_2D()
        bg_value_2D = self.get_bg_value_2D(data_2D.shape)

        self.axes_model.set_data(axes)
        self.hlabels_model.set_data(hlabels)
        self.vlabels_model.set_data(vlabels)
        # using the protected version of the method to avoid calling reset() several times
        self.data_model._set_data(data_2D)
        self.data_model._set_changes(changes_2D)
        self.data_model._set_bg_value(bg_value_2D)

    def get_data(self):
        return self.la_data

    @property
    def ndim(self):
        return self.filtered_data.ndim

    @property
    def dtype(self):
        return self.la_data.dtype

    def update_changes(self):
        for k, v in self.data_model.changes.items():
            self.changes[self._map_filtered_to_global(k)] = v

    def _map_filtered_to_global(self, k):
        """
        map local (filtered data) 2D key to global (unfiltered) ND key.

        Parameters
        ----------
        k: tuple
            Positional index (row, column) of the modified data cell.

        Returns
        -------
        tuple
            Labels associated with the modified element of the non-filtered array.
        """
        # transform local positional index key to (axis_ids: label) dictionary key.
        # Contains only displayed axes
        row, col = k
        labels = [self.filtered_data.axes[-1].labels[col]]
        for axis in reversed(self.filtered_data.axes[:-1]):
            row, position = divmod(row, len(axis))
            labels = [axis.labels[position]] + labels
        axes_ids = list(self.filtered_data.axes.ids)
        dkey = dict(zip(axes_ids, labels))
        # add the "scalar" parts of the filter to it (ie the parts of the
        # filter which removed dimensions)
        dkey.update({k: v for k, v in self.current_filter.items() if np.isscalar(v)})
        # re-transform it to tuple (to make it hashable/to store it in .changes)
        return tuple(dkey[axis_id] for axis_id in self.la_data.axes.ids)

    def _map_global_to_filtered(self, k):
        """
        map global (unfiltered) ND key to local (filtered) 2D key

        Parameters
        ----------
        k: tuple
            Labels associated with the modified element of the non-filtered array.

        Returns
        -------
        tuple
            Positional index (row, column) of the modified data cell.
        """
        assert isinstance(k, tuple) and len(k) == self.la_data.ndim
        dkey = {axis_id: axis_key for axis_key, axis_id in zip(k, self.la_data.axes.ids)}
        # transform global dictionary key to "local" (filtered) key by removing
        # the parts of the key which are redundant with the filter
        for axis_id, axis_filter in self.current_filter.items():
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
            index_key = self.filtered_data._translated_key(dkey)
        except ValueError:
            return None
        # transform positional ND key to positional 2D key
        strides = np.append(1, np.cumprod(self.filtered_data.shape[1:-1][::-1], dtype=int))[::-1]
        return (index_key[:-1] * strides).sum(), index_key[-1]

    def change_filter(self, axis, indices):
        # must be done before changing self.current_filter
        self.update_changes()
        axis_id = self.la_data.axes.axis_id(axis)
        if not indices or len(indices) == len(axis.labels):
            if axis_id in self.current_filter:
                del self.current_filter[axis_id]
        else:
            if len(indices) == 1:
                self.current_filter[axis_id] = axis.labels[indices[0]]
            else:
                self.current_filter[axis_id] = axis.labels[indices]
        self.update_filtered_data()
        self.data_model.reset()

    def clear_changes(self):
        self.changes.clear()
        self.data_model.changes.clear()

    def accept_changes(self):
        """Accept changes"""
        # update changes
        self.update_changes()
        # update internal data
        axes = self.la_data.axes
        for k, v in self.changes.items():
            self.la_data.i[axes.translate_full_key(k)] = v
        # update models
        self.update_filtered_data()
        self.data_model.reset()
        # clear changes
        self.clear_changes()
        # return modified data
        return self.la_data

    def reject_changes(self):
        """Reject changes"""
        # clear changes
        self.clear_changes()
        self.data_model.reset_minmax()
        self.data_model.reset()
