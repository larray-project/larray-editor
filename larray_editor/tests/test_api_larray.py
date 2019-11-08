from __future__ import absolute_import, division, print_function

"""Array editor test"""

import logging

import numpy as np
import larray as la

from larray_editor.api import *
from larray_editor.utils import logger

logger.setLevel(logging.DEBUG)

lipro = la.Axis(['P%02d' % i for i in range(1, 16)], 'lipro')
age = la.Axis('age=0..115')
sex = la.Axis('sex=M,F')

vla = 'A11,A12,A13,A23,A24,A31,A32,A33,A34,A35,A36,A37,A38,A41,A42,A43,A44,A45,A46,A71,A72,A73'
wal = 'A25,A51,A52,A53,A54,A55,A56,A57,A61,A62,A63,A64,A65,A81,A82,A83,A84,A85,A91,A92,A93'
bru = 'A21'
# list of strings
belgium = la.union(vla, wal, bru)

geo = la.Axis(belgium, 'geo')

# arr1 = la.ndtest((sex, lipro))
# edit(arr1)

# data2 = np.arange(116 * 44 * 2 * 15).reshape(116, 44, 2, 15) \
#           .astype(float)
# data2 = np.random.random(116 * 44 * 2 * 15).reshape(116, 44, 2, 15) \
#           .astype(float)
# data2 = (np.random.randint(10, size=(116, 44, 2, 15)) - 5) / 17
# data2 = np.random.randint(10, size=(116, 44, 2, 15)) / 100 + 1567
# data2 = np.random.normal(51000000, 10000000, size=(116, 44, 2, 15))
arr2 = la.random.normal(axes=(age, geo, sex, lipro))
# arr2 = la.ndrange([100, 100, 100, 100, 5])
# arr2 = arr2['F', 'A11', 1]

# view(arr2[0, 'A11', 'F', 'P01'])
# view(arr1)
# view(arr2[0, 'A11'])
# edit(arr1)
# print(arr2[0, 'A11', :, 'P01'])
# edit(arr2.astype(int), minvalue=-99, maxvalue=55.123456)
# edit(arr2.astype(int), minvalue=-99)
# arr2.i[0, 0, 0, 0] = np.inf
# arr2.i[0, 0, 1, 1] = -np.inf
# arr2 = [0.0000111, 0.0000222]
# arr2 = [0.00001, 0.00002]
# edit(arr2, minvalue=-99, maxvalue=25.123456)
# print(arr2[0, 'A11', :, 'P01'])

# arr2 = la.random.normal(0, 10, axes="d0=0..4999;d1=0..19")
# edit(arr2)

# view(['a', 'bb', 5599])
# view(np.arange(12).reshape(2, 3, 2))
# view([])

data3 = np.random.normal(0, 1, size=(2, 15))
arr3 = la.ndtest((30, sex))
# data4 = np.random.normal(0, 1, size=(2, 15))
# arr4 = la.Array(data4, axes=(sex, lipro))

# arr4 = arr3.copy()
# arr4['F'] /= 2
arr4 = arr3.min(sex)
arr5 = arr3.max(sex)
arr6 = arr3.mean(sex)

# test isssue #35
arr7 = la.from_lists([['a',                   1,                    2,                    3],
                   [ '', 1664780726569649730, -9196963249083393206, -7664327348053294350]])


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


demo = make_demo(9, 2.5, 1.5)
sphere = make_sphere(9, 4)
extreme_array = la.Array([-la.inf, -1, 0, la.nan, 1, la.inf])
scalar = la.Array(0)
arr_empty = la.Array([])
arr_empty_2d = la.Array([[], []])
arr_obj = la.ndtest((2, 3)).astype(object)
arr_str = la.ndtest((2, 3)).astype(str)
big = la.ndtest((1000, 1000, 500))
big1d = la.ndtest(1000000)
# force big1d.axes[0]._mapping to be created so that we do not measure that delay in the editor
big1d[{}]

# test autoresizing
long_labels = la.zeros('a=a_long_label,another_long_label; b=this_is_a_label,this_is_another_one')
long_axes_names = la.zeros('first_axis=a0,a1; second_axis=b0,b1')

# compare(arr3, arr4, arr5, arr6)

# view(stack((arr3, arr4), la.Axis('arrays=arr3,arr4')))
# ses = la.Session(arr2=arr2, arr3=arr3, arr4=arr4, arr5=arr5, arr6=arr6, arr7=arr7, long_labels=long_labels,
#                  long_axes_names=long_axes_names, data2=data2, data3=data3)

# from larray.tests.common import abspath
# file = abspath('test_session.xlsx')
# ses.save(file)

# import cProfile as profile
# profile.runctx('edit(Session(arr2=arr2))', vars(), {},
#                'c:\\tmp\\edit.profile')
debug()
edit()
# edit(ses)
# edit(file)
# edit('fake_path')
# edit(REOPEN_LAST_FILE)

edit(arr2)

compare(arr3, arr3 + 1.0)
compare(np.random.normal(0, 1, size=(10, 2)), np.random.normal(0, 1, size=(10, 2)))
compare(la.Session(arr4=arr4, arr3=arr3, data=data3),
        la.Session(arr4=arr4 + 1.0, arr3=arr3 * 2.0, data=data3 * 1.05))
# compare(la.Session(arr2=arr2, arr3=arr3),
#         la.Session(arr2=arr2 + 1.0, arr3=arr3 * 2.0))

# s = la.local_arrays()
# view(s)
# print('HDF')
# s.save('x.h5')
# print('\nEXCEL')
# s.save('x.xlsx')
# print('\nCSV')
# s.save('x_csv')
# print('\n open HDF')
# edit('x.h5')
# print('\n open EXCEL')
# edit('x.xlsx')
# print('\n open CSV')
# edit('x_csv')

arr1 = la.ones((geo, sex))
arr2 = la.random.normal(axes=(geo, sex))
compare(arr1, arr2, atol=0.5)
compare(arr1, arr2, rtol=0.3)

arr2 = la.where(arr2 > 1, arr1, -arr1)
arr1['M'] = la.nan
arr2['M'] = la.nan
compare(arr1, arr2, nans_equal=False)

arr1 = la.ndtest((3, 3))
arr2 = 2 * arr1
arr3 = la.where(arr1 % 2 == 0, arr1, -arr1)
compare(arr1, arr2, arr3, bg_gradient='blue-red')


def test_run_editor_on_exception(local_arr1):
    return arr2['my_invalid_key']

run_editor_on_exception()
# run_editor_on_exception(usercode_traceback=False)
# run_editor_on_exception(usercode_traceback=False, usercode_frame=False)

test_run_editor_on_exception(arr1)
