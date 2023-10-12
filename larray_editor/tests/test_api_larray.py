"""Array editor test"""

import array
import logging
from collections import OrderedDict, namedtuple
from pathlib import Path

import numpy as np
import qtpy
import larray as la
import pandas as pd


from larray_editor.api import edit
# from larray_editor.api import view, edit, debug, compare
from larray_editor.utils import logger

print(f"Using {qtpy.API_NAME} as Qt API")
logger.setLevel(logging.DEBUG)

# array objects
array_double = array.array('d', [1.0, 2.0, 3.14])
array_signed_int = array.array('l', [1, 2, 3, 4, 5])
array_signed_int_empty = array.array('l')
array_unicode = array.array('u', 'hello \u2641')

# list
list_empty = []
list_int = [2, 5, 7, 3]
list_mixed = ['abc', 1.1, True, 1.0, 42, [1, 2]]
list_seq_mixed = [[1], [2, 3, 4], [5, 6]]
list_seq_regular = [[1, 2], [3, 4], [5, 6]]

# tuple
tuple_empty = ()
tuple_int = (2, 5, 7, 3)
tuple_mixed = ('abc', 1.1, True, 1.0, 42, (1, 2))
tuple_seq_mixed = ((1,), (2, 3, 4), (5, 6))
tuple_seq_regular = ((1, 2), (3, 4), (5, 6))

# named tuple
PersonNamedTuple = namedtuple('Person', ['name', 'age', 'male', 'height'])
namedtuple1 = PersonNamedTuple("name1", age=42, male=True, height=1.80)
namedtuple2 = PersonNamedTuple("name2", age=41, male=False, height=1.76)

# set
set_int = {2, 4, 7, 3}
set_int_big = set(range(10 ** 7))
set_mixed = {2, "hello", 7.0, True}
set_str = {"a", "b", "c", "d"}

# dict
dict_str_int = {"a": 2, "b": 5, "c": 7, "d": 3}
dict_int_int = {0: 2, 2: 4, 5: 7, 1: 3}
dict_int_str = {0: "a", 2: "b", 5: "c", 1: "d"}
dict_str_mixed = {"a": 2, "b": "hello", "c": 7.0, "d": True}

# dict views
dictview_keys = dict_str_mixed.keys()
dictview_items = dict_str_mixed.items()
dictview_values = dict_str_mixed.values()

# OrderedDict
odict_int_int = OrderedDict(dict_int_int)
odict_int_str = OrderedDict(dict_int_str)
odict_str_int = OrderedDict(dict_str_int)
odict_str_mixed = OrderedDict(dict_str_mixed)

# numpy arrays
np_arr0d = np.full((), 42, dtype=float)
np_arr1d = np.random.normal(0, 1, size=100)
np_arr1d_empty = np.random.normal(0, 1, size=0)
np_arr2d = np.random.normal(0, 1, size=(100, 100))
np_arr2d_0col = np.random.normal(0, 1, size=(10, 0))
np_arr2d_0row = np.random.normal(0, 1, size=(0, 10))
np_arr3d = np.random.normal(0, 1, size=(10, 10, 10))
np_dtype = np.dtype([('name', '<U11'), ('age', int), ('male', bool), ('height', float)])
np_struct_arr = np.array([('name1', 42,  True, 1.80),
                          ('name2', 41, False, 1.76),
                          ('name3', 43, False, 1.78),
                          ('name4', 44,  True, 1.77)], dtype=np_dtype)
np_struct_arr_2d = np.array([[('name1', 41,  True, 1.80),
                              ('name2', 42, False, 1.79)],
                             [('name3', 43, False, 1.78),
                              ('name4', 44,  True, 1.77)]], dtype=np_dtype)


def make_circle(width=20, radius=9):
    x, y = la.Axis(width, 'x'), la.Axis(width, 'y')
    center = (width - 1) / 2
    return la.maximum(radius - la.sqrt((x - center) ** 2 + (y - center) ** 2), 0)


def make_sphere(width=20, radius=9):
    x, y, z = la.Axis(width, 'x'), la.Axis(width, 'y'), la.Axis(width, 'z')
    center = (width - 1) / 2
    return la.maximum(radius - la.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2), 0)


def make_demo(width=20, ball_radius=5, path_radius=5, steps=30):
    x, y = la.Axis(width, 'x'), la.Axis(width, 'y')
    t = la.Axis(steps, 't')
    center = (width - 1) / 2
    ball_center_x = la.sin(la.radians(t * 360 / steps)) * path_radius + center
    ball_center_y = la.cos(la.radians(t * 360 / steps)) * path_radius + center
    return la.maximum(ball_radius - la.sqrt((x - ball_center_x) ** 2 + (y - ball_center_y) ** 2), 0).transpose(x, y)


def test_matplotlib_show_interaction():
    import matplotlib.pyplot as plt

    arr = la.ndtest((3, 4))
    arr.plot()
    plt.show()
    edit()


lipro = la.Axis('lipro=P01..P15')
age = la.Axis('age=0..29')
sex = la.Axis('sex=M,F')
geo = la.Axis(['A11', 'A25', 'A51', 'A21'], 'geo')

la_arr2 = la.random.normal(axes=(age, geo, sex, lipro))
la_arr3 = la.ndtest((30, sex))
la_arr4 = la_arr3.min(sex)
la_arr5 = la_arr3.max(sex)
la_arr6 = la_arr3.mean(sex)

# test isssue #35
la_arr7 = la.from_lists([['a',                   1,                    2,                    3],
                         [ '', 1664780726569649730, -9196963249083393206, -7664327348053294350]])

la_demo = make_demo(9, 2.5, 1.5)
la_sphere = make_sphere(9, 4)
la_extreme_array = la.Array([-la.inf, -1, 0, la.nan, 1, la.inf])
la_scalar = la.Array(0)
la_all_nan = la.Array([la.nan, la.nan])
# FIXME: this test should be updated for buffer
# this is crafted so that the entire 500 points sample is all nan but
# other values need coloring
la_full_buffer_nan_should_not_be_all_white = la.ndtest(1000, dtype=float)
la_full_buffer_nan_should_not_be_all_white['a0'::2] = la.nan
la_empty = la.Array([])
la_empty_2d = la.Array([[], []])
la_obj_numeric = la.ndtest((2, 3)).astype(object)
la_boolean = (la_arr3 % 3) == 0
la_obj_mixed = la.ndtest((2, 3)).astype(object)
la_obj_mixed['a0', 'b1'] = 'hello'
la_str = la.ndtest((2, 3)).astype(str)
la_big = la.ndtest((1000, 1000, 500))
la_big1d = la.ndtest(1000000)
# force big1d.axes[0]._mapping to be created so that we do not measure that delay in the editor
_ = la_big1d[{}]
del _

# test auto-resizing
la_long_labels = la.zeros('a=a_long_label,another_long_label; b=this_is_a_label,this_is_another_one')
la_long_axes_names = la.zeros('first_axis=a0,a1; second_axis=b0,b1')

# compare(la_arr3, la_arr4, la_arr5, la_arr6)

# view(stack((arr3, arr4), la.Axis('arrays=arr3,arr4')))
# ses = la.Session(arr2=arr2, arr3=arr3, arr4=arr4, arr5=arr5, arr6=arr6, arr7=arr7, long_labels=long_labels,
#                  long_axes_names=long_axes_names, data2=data2, data3=data3)

# from larray.tests.common import abspath
# file = abspath('test_session.xlsx')
# ses.save(file)

# import cProfile as profile
# profile.runctx('edit(Session(arr2=arr2))', vars(), {},
#                'c:\\tmp\\edit.profile')
# debug()
# edit()
# edit(Path('../test_object.h5'))
# edit(ses)
# edit(file)
# edit('fake_path')
# edit(REOPEN_LAST_FILE)

# edit(arr2)

# test issue #247 (same names)
# compare(arr3, arr3)
# compare(arr3, arr3 + 1.0)
# compare(arr3, arr3 + 1.0, names=['arr3', 'arr3 + 1.0'])
# compare(np.random.normal(0, 1, size=(10, 2)), np.random.normal(0, 1, size=(10, 2)))

# sess1 = la.Session(arr4=arr4, arr3=arr3, data=data3)
# sess1.save('sess1.h5')
# sess2 = la.Session(arr4=arr4 + 1.0, arr3=arr3 * 2.0, data=data3 * 1.05)
# compare('sess1.h5', sess2)   # sess1.h5/data is nan because np arrays are not saved to H5
# compare(Path('sess1.h5'), sess2)
# compare(la.Session(arr2=arr2, arr3=arr3),
#         la.Session(arr2=arr2 + 1.0, arr3=arr3 * 2.0))

# s = la.local_arrays()
# view(s)
# s.save('x.h5')
# s.save('x.xlsx')
# s.save('x_csv')
# print('\n open HDF')
# edit('x.h5')
# print('\n open EXCEL')
# edit('x.xlsx')
# print('\n open CSV')
# edit('x_csv')

# arr1 = la.ones((geo, sex))
# arr2 = la.random.normal(axes=(geo, sex))
# compare(arr1, arr2, atol=0.5)
# compare(arr1, arr2, rtol=0.3)
#
# arr2 = la.where(arr2 > 1, arr1, -arr1)
# arr1_m = arr1['M'].copy()
# arr1['M'] = la.nan
# compare(arr1, arr2, nans_equal=False, title='with nans on left side')
# arr2['M'] = la.nan
# compare(arr1, arr2, nans_equal=False, title='with nans on both sides')
# arr1['M'] = arr1_m
# compare(arr1, arr2, nans_equal=False, title='with nans on right side')
#
# arr1 = la.ndtest((3, 3))
# arr2 = 2 * arr1
# arr3 = la.where(arr1 % 2 == 0, arr1, -arr1)
# compare(arr1, arr2, arr3, bg_gradient='blue-white-red')

# test for arr.plot(show=True) which is the default
# =================================================
# arr = la.ndtest((20, 5)) + la.random.randint(0, 3, axes="a=a0..a19;b=b0..b4")
# arr.plot(animate='a')


def test_run_editor_on_exception(local_arr):
    return local_arr['my_invalid_key']


# run_editor_on_exception()
# run_editor_on_exception(usercode_traceback=False)
# run_editor_on_exception(usercode_traceback=False, usercode_frame=False)

# test_run_editor_on_exception(arr2)

pd_df_mixed = pd.DataFrame(np_struct_arr)
pd_df1 = la_arr3.df
pd_df2 = la_arr2.df
pd_df3 = pd_df2.T
pd_df4 = pd_df2.unstack()
pd_df_str = pd_df2.astype(str)
pd_series = pd_df2.stack()

path_dir = Path('.')
path_py = Path('test_adapter.py')
path_csv = Path('be.csv')

try:
    import pyarrow as pa

    pyarrow_int_array = pa.array([2, 4, 5, 42])
    pyarrow_str_array = pa.array(["Hello", "from", "Arrow", "!"])
    pyarrow_table = pa.Table.from_arrays([pyarrow_int_array, pyarrow_str_array],
                                         names=["int_col", "str_col"])
except ImportError:
    pass

# import cProfile as profile
# profile.runctx('edit(la.Session(arr2=arr2))', vars(), {},
#                'c:/tmp/edit.profile')
import pstats
pstats_stats = pstats.Stats('c:\\tmp\\edit.profile')

edit()
# debug()

test_matplotlib_show_interaction()
