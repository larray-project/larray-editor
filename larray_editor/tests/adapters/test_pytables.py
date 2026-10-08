import larray as la
import numpy as np
import pandas as pd
import pytest
from larray.tests.common import needs_pytables

from larray_editor.tests.adapters.common import hdf_adapter_values


@needs_pytables
def test_pytables_pandas_frame_mixed_type_index(tmp_path):
    index = pd.Index(['x', 1], dtype=object, name='i')
    with pytest.warns(pd.errors.PerformanceWarning, match='pickle object types'):
        rows = hdf_adapter_values(pd.DataFrame({'c': [1, 2]}, index=index), tmp_path)
    assert rows == [['i', 'c'],
                    ['x', 1],
                    [1, 2]]


@needs_pytables
def test_pytables_pandas_series(tmp_path):
    series = pd.Series([1.5, 2.5], index=pd.Index(['x', 'y'], name='i'))
    assert hdf_adapter_values(series, tmp_path) == [['i', ''],
                                                    ['x', 1.5],
                                                    ['y', 2.5]]

    series = pd.Series(['u', 'v'])
    assert hdf_adapter_values(series, tmp_path) == [[None, ''],
                                                    [0, 'u'],
                                                    [1, 'v']]

    index = pd.MultiIndex.from_tuples([('x', 1), ('y', 2)], names=['i', 'j'])
    series = pd.Series([1, 2], index=index)
    assert hdf_adapter_values(series, tmp_path) == [['i', 'j', ''],
                                                    ['x', 1, 1],
                                                    ['y', 2, 2]]


@needs_pytables
def test_pytables_larray_axis_and_group(tmp_path):
    axis = la.Axis('a=a0..a2')
    assert hdf_adapter_values(axis, tmp_path) == [[None, ''],
                                                  [0, 'a0'],
                                                  [1, 'a1'],
                                                  [2, 'a2']]

    bytes_axis = la.Axis(np.array([b'b0', b'b1']), 'b')
    assert hdf_adapter_values(bytes_axis, tmp_path) == [[None, ''],
                                                        [0, b'b0'],
                                                        [1, b'b1']]

    group = axis['a0,a2'] >> 'group'
    assert hdf_adapter_values(group, tmp_path) == [[None, ''],
                                                   [0, 'a0'],
                                                   [1, 'a2']]

    # stored as labels (bytes)
    slice_group = axis['a1':'a2'] >> 'slice_group'
    assert hdf_adapter_values(slice_group, tmp_path) == [[None, ''],
                                                         [0, 'a1'],
                                                         [1, 'a2']]
