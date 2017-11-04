from __future__ import absolute_import, division, print_function

import numpy as np
import larray as la
from larray_editor.utils import Product, _LazyNone, _LazyDimLabels


class LArrayDataAdapter(object):
    def __init__(self, axes_model, hlabels_model, vlabels_model, data_model, data=None,
                 changes=None, current_filter=None, nb_dims_hlabels=1, bg_gradient=None, bg_value=None):
        # set models
        self.axes_model = axes_model
        self.hlabels_model = hlabels_model
        self.vlabels_model = vlabels_model
        self.data_model = data_model
        # set number of dims of hlabels
        self.nb_dims_hlabels = nb_dims_hlabels
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
        self.set_data(data, bg_value, current_filter)

    def set_changes(self, changes=None):
        assert isinstance(changes, dict)
        self.changes = changes

    def update_nb_dims_hlabels(self, nb_dims_hlabels):
        self.nb_dims_hlabels = nb_dims_hlabels
        self.update_axes_and_labels()

    def get_axes_names(self):
        return self.filtered_data.axes.display_names

    def get_axes(self):
        axes_names = self.filtered_data.axes.display_names
        # test self.filtered_data.size == 0 is required in case an instance built as LArray([]) is passed
        # test len(axes) == 0 is required when a user filters until to get a scalar
        if self.filtered_data.size == 0 or len(axes_names) == 0:
            return None
        elif len(axes_names) == 1:
            return [axes_names]
        else:
            nb_dims_vlabels = len(axes_names) - self.nb_dims_hlabels
            # axes corresponding to horizontal labels are set to the last column
            res = [['' for c in range(nb_dims_vlabels-1)] + [axis_name] for axis_name in axes_names[nb_dims_vlabels:]]
            # axes corresponding to vertical labels are set to the last row
            res = res + [[axis_name for axis_name in axes_names[:nb_dims_vlabels]]]
            return res

    def get_labels(self):
        axes = self.filtered_data.axes
        if self.filtered_data.size == 0 or len(axes) == 0:
            vlabels = None
            hlabels = None
        else:
            nb_dims_vlabels = len(axes) - self.nb_dims_hlabels
            def get_labels_product(axes, extra_row=False):
                if len(axes) == 0:
                    return [[' ']]
                else:
                    # XXX: appends a fake axis instead of using _LazyNone because
                    # _LazyNone mess up with LabelsArrayModel.get_values (in which slices are used)
                    if extra_row:
                        axes.append(la.Axis([' ']))
                    prod = Product(axes.labels)
                    return [_LazyDimLabels(prod, i) for i in range(len(axes.labels))]
            vlabels = get_labels_product(axes[:nb_dims_vlabels])
            hlabels = get_labels_product(axes[nb_dims_vlabels:], nb_dims_vlabels > 0)
        return vlabels, hlabels

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
    def set_data(self, data, bg_value=None, current_filter=None):
        if data is None:
            data = la.LArray([])
        if current_filter is None:
            self.current_filter = {}
        self.changes = {}
        self.la_data = la.aslarray(data)
        self.bg_value = la.aslarray(bg_value) if bg_value is not None else None
        self.update_filtered_data(current_filter, reset_minmax=True)

    def update_axes_and_labels(self):
        axes = self.get_axes()
        vlabels, hlabels = self.get_labels()
        self.axes_model.set_data(axes)
        self.hlabels_model.set_data(hlabels)
        self.vlabels_model.set_data(vlabels)

    def update_data_2D(self, reset_minmax=False):
        data_2D = self.get_2D_data()
        changes_2D = self.get_changes_2D()
        bg_value_2D = self.get_bg_value_2D(data_2D.shape)
        self.data_model.set_data(data_2D, changes_2D, reset_minmax=reset_minmax)
        self.data_model.set_bg_value(bg_value_2D)

    def update_filtered_data(self, current_filter=None, reset_minmax=False):
        if current_filter is not None:
            assert isinstance(current_filter, dict)
            self.current_filter = current_filter
        self.filtered_data = self.la_data[self.current_filter]
        if np.isscalar(self.filtered_data):
            self.filtered_data = la.aslarray(self.filtered_data)
        self.update_axes_and_labels()
        self.update_data_2D(reset_minmax=reset_minmax)

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
