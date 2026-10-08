import numpy as np
import pytest

from larray_editor.arrayadapter import AbstractAdapter, get_adapter

try:
    import polars
except ImportError:
    polars = None
try:
    import pyarrow
except ImportError:
    pyarrow = None


needs_polars = (
    pytest.mark.skipif(polars is None,
                       reason="polars is required for this test")
)
needs_pyarrow = (
    pytest.mark.skipif(pyarrow is None,
                       reason="pyarrow is required for this test")
)


def adapter_values(obj_or_adapter):
    """
    Return the labels and values an adapter provides to display an object.

    obj_or_adapter can be an adapter or an object (to display with its adapter).
    The first row contains the index names followed by the column labels, the other rows
    contain the row labels followed by the values. Values are converted to Python scalars
    (using tolist), so this cannot test how values are displayed (formatted).
    """
    if isinstance(obj_or_adapter, AbstractAdapter):
        adapter = obj_or_adapter
    else:
        adapter = get_adapter(obj_or_adapter)
    nrows, ncols = adapter.shape2d()
    (hlabels,) = np.asarray(adapter.get_hlabels_values(0, ncols)).tolist()
    vlabels = np.asarray(adapter.get_vlabels_values(0, nrows)).tolist()
    values = adapter.get_data_values_and_attributes(0, 0, ncols, nrows)['values'].tolist()
    # zip (below) would silently hide a shape which does not match the data
    assert len(hlabels) == ncols and len(vlabels) == len(values) == nrows
    rows = [list(adapter.get_vnames()) + hlabels]
    rows += [row_labels + row_values for row_labels, row_values in zip(vlabels, values)]
    if adapter is not obj_or_adapter:
        adapter.close()
    return rows


def hdf_adapter_values(obj, tmp_path):
    """Store obj in an HDF file and return adapter_values for it"""
    import tables

    fpath = tmp_path / 'test.h5'
    obj.to_hdf(fpath, key='obj')
    with tables.open_file(fpath) as f:
        rows = adapter_values(f.root.obj)
    fpath.unlink()
    return rows
