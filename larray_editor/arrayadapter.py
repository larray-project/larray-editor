from __future__ import absolute_import, division, print_function

import numpy as np
import larray as la
from larray_editor.utils import Product, _LazyNone, _LazyDimLabels


class LArrayDataAdapter(object):
    def __init__(self, axes_model, xlabels_model, ylabels_model, data_model,
                 data=None, changes=None, current_filter=None, bg_gradient=None, bg_value=None):
        # set models
        self.axes_model = axes_model
        self.xlabels_model = xlabels_model
        self.ylabels_model = ylabels_model
        self.data_model = data_model
        # set current filter
        if current_filter is None:
            current_filter = {}
        assert isinstance(current_filter, dict)
        self.current_filter = current_filter
        # set changes
        if changes is None:
            changes = {}
        self.set_changes(changes)
        # set data
        if data is None:
            data = np.empty((0, 0), dtype=np.int8)
        self.set_data(data, bg_gradient, bg_value)

    def set_changes(self, changes=None):
        assert isinstance(changes, dict)
        self.changes = changes

    def get_axes_names(self):
        return self.filtered_data.axes.display_names

    def get_axes(self):
        axes_names = self.filtered_data.axes.display_names
        if len(axes_names) >= 2:
            axes_names = axes_names[:-2] + [axes_names[-2] + '\\' + axes_names[-1]]
        return [[axis_name] for axis_name in axes_names]

    def get_xlabels(self):
        axes = self.filtered_data.axes
        if len(axes) == 0:
            return [[]]
        elif len(axes.labels[-1]) == 0:
            return [['']]
        else:
            return [[label] for label in axes.labels[-1]]

    def get_ylabels(self):
        axes = self.filtered_data.axes
        if len(axes) == 0:
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

    # XXX: or create two methods?:
    # - set_data (which reset the current filter)
    # - update_data (which sets new data but keeps current filter unchanged)
    def set_data(self, data, bg_gradient=None, bg_value=None, current_filter=None):
        if data is None:
            data = la.LArray([])
        if current_filter is None:
            self.current_filter = {}
        self.la_data = la.aslarray(data)
        self.update_filtered_data(current_filter)
        self.data_model.set_background(bg_gradient, bg_value)

    def update_filtered_data(self, current_filter=None):
        if current_filter is not None:
            assert isinstance(current_filter, dict)
            self.current_filter = current_filter
        self.filtered_data = self.la_data[self.current_filter]
        if np.isscalar(self.filtered_data):
            self.filtered_data = la.aslarray(self.filtered_data)
        if len(self.filtered_data) == 0:
            axes = [[]]
            xlabels = [[]]
            ylabels = [[]]
        else:
            axes = self.get_axes()
            xlabels = self.get_xlabels()
            ylabels = self.get_ylabels()
        data_2D = self.get_2D_data()
        changes_2D = self.get_changes_2D()
        self.axes_model.set_data(axes)
        self.xlabels_model.set_data(xlabels)
        self.ylabels_model.set_data(ylabels)
        self.data_model.set_data(data_2D, changes_2D)

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
        return self.filtered_data

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
