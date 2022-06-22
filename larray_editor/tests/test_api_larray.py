"""Array editor test"""

import logging
# from pathlib import Path

import qtpy
import larray as la

from larray_editor.api import edit
# from larray_editor.api import view, edit, debug, compare
from larray_editor.utils import logger

print(f"Using {qtpy.API_NAME} as Qt API")
logger.setLevel(logging.DEBUG)


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
edit()
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

# debug()

test_matplotlib_show_interaction()
