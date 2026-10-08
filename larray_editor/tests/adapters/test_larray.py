import larray as la

from larray_editor.tests.adapters.common import adapter_values


def test_larray_empty_axis():
    a = la.Axis('a=a0..a1')
    b = la.Axis('b=b0..b2')
    len0_axis = la.Axis([], 'len0_axis')

    zero_rows = la.zeros((len0_axis, b))
    assert adapter_values(zero_rows) == [['len0_axis', 'b0', 'b1', 'b2']]
    zero_cols = la.zeros((a, len0_axis))
    assert adapter_values(zero_cols) == [['a'],
                                         ['a0'],
                                         ['a1']]
