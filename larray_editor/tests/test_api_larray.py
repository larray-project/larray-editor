"""Array editor test"""

import importlib
import array
import logging
import sys
import zipfile
from collections import OrderedDict, namedtuple
import sqlite3
from pathlib import Path

import numpy as np
import qtpy
import larray as la
import pandas as pd

from larray_editor.api import edit
# from larray_editor.api import view, edit, debug, compare
from larray_editor.utils import logger

# Configure logging to output messages to the console
logging.basicConfig(
    # Show warnings and above for all loggers
    level=logging.WARNING,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout
)
# Set our own logger to DEBUG
logger.setLevel(logging.DEBUG)
logger.info(f"Using {qtpy.API_NAME} as Qt API")

# array objects
array_double = array.array('d', [1.0, 2.0, 3.14])
array_signed_int = array.array('l', [1, 2, 3, 4, 5])
array_signed_int_empty = array.array('l')
# should show as hello alpha and omega
array_unicode = array.array('w', 'hello \u03B1 and \u03C9')

# list
list_empty = []
list_int = [2, 5, 7, 3]
list_mixed = ['abc', 1.1, True, 1.0, 42, [1, 2]]
list_seq_mixed = [[1], [2, 3, 4], [5, 6]]
list_seq_regular = [[1, 2], [3, 4], [5, 6]]
list_unicode = ["\N{grinning face}", "\N{winking face}"]
list_mixed_tuples = [
    ("C", 1972),
    ("Fortran", 1957),
    ("Python", 1991),
    ("Go", 2009),
]

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
np_big1d = np.arange(1000 * 1000 * 500)
np_big3d = np_big1d.reshape((1000, 1000, 500))
np_dtype = np.dtype([('name', '<U11'), ('age', int), ('male', bool), ('height', float)])
np_struct_arr = np.array([('name1', 42,  True, 1.80),
                          ('name2', 41, False, 1.76),
                          ('name3', 43, False, 1.78),
                          ('name4', 44,  True, 1.77)], dtype=np_dtype)
np_struct_arr_2d = np.array([[('name1', 41,  True, 1.80),
                              ('name2', 42, False, 1.79)],
                             [('name3', 43, False, 1.78),
                              ('name4', 44,  True, 1.77)]], dtype=np_dtype)
mv_arr1d = memoryview(np_arr1d)
mv_arr2d = memoryview(np_arr2d)
mv_arr3d = memoryview(np_arr3d)
mv_big3d = memoryview(np_big3d)

# these are not supported
# mv_struct_arr = memoryview(np_struct_arr)
# mv_struct_arr_2d = memoryview(np_struct_arr_2d)


def make_demo(width=20, ball_radius=5, path_radius=5, steps=30):
    x, y = la.Axis(width, 'x'), la.Axis(width, 'y')
    t = la.Axis(steps, 't')
    center = (width - 1) / 2
    ball_center_x = la.sin(la.radians(t * 360 / steps)) * path_radius + center
    ball_center_y = la.cos(la.radians(t * 360 / steps)) * path_radius + center
    return la.maximum(ball_radius - la.sqrt((x - ball_center_x) ** 2 + (y - ball_center_y) ** 2), 0).transpose(x, y)


def test_edit_after_matplotlib_show():
    import matplotlib.pyplot as plt

    arr = la.ndtest((3, 4))
    arr.plot()
    plt.show()
    edit()


# this needs to be called in the interactive console and should open a single plot window,
# not two (see issue #265)
def test_plot_returning_ax_and_using_show():
    import matplotlib.pyplot as plt

    arr = la.ndtest(4)
    ax = arr.plot()
    plt.show()
    return ax


lipro = la.Axis('lipro=P01..P15')
age = la.Axis('age=0..29')
sex = la.Axis('sex=M,F')
geo = la.Axis(['A11', 'A25', 'A51', 'A21'], 'geo')

la_float_4d_many_digits = la.random.normal(axes=(age, geo, sex, lipro))
la_float_4d_many_digits['P01', 'A11', 0] = la.nan
la_int_1d = la.ndtest(age)
# FIXME: the new align code makes this fail
# la_int2d = la.ndtest((30, sex))
la_int_2d = la.ndtest((age, sex))
la_float_round_values = la_int_2d.mean(sex)

# test isssue #35
la_very_large_ints = la.from_lists([
    ['a',                   1,                    2,                    3],
    [ '', 1664780726569649730, -9196963249083393206, -7664327348053294350]
])

la_demo = make_demo(9, 2.5, 1.5)
la_extreme_array = la.Array([-la.inf, -1, 0, la.nan, 1, la.inf])
la_scalar = la.Array(0)
la_all_nan = la.Array([la.nan, la.nan])

# this is crafted so that the entire byffer is all nan but
# other values need coloring
la_full_buffer_nan_should_not_be_all_white = la.ndtest((100, 100), dtype=float)
la_full_buffer_nan_should_not_be_all_white[:'a50',:'b50'] = la.nan
la_empty = la.Array([])
la_empty_2d = la.Array([[], []])
la_obj_numeric = la.ndtest((2, 3)).astype(object)
la_boolean = (la_int_2d % 3) == 0
la_obj_mixed = la.ndtest((2, 3)).astype(object)
la_obj_mixed['a0', 'b1'] = 'Hello, this is a string !'
la_obj_mixed['a0', 'b2'] = 1.95
la_obj_mixed['a1', 'b0'] = True
la_str = la.ndtest((2, 3)).astype(str)

la_big1d = la.Array(np_big1d[:1000_000], 'a=a0..a999999')
la_big3d = la.Array(np_big3d, 'a=a0..a999;b=b0..b999;c=c0..c499')
# force big1d.axes[0]._mapping to be created so that we do not measure that delay in the editor
# _ = la_big1d[{}]
# del _

# test auto-resizing
la_long_labels = la.zeros('a=a_long_label,another_long_label; b=this_is_a_label,this_is_another_one')
la_long_axes_names = la.zeros('first_axis=a0,a1; second_axis=b0,b1')

if importlib.util.find_spec('xlwings') is not None:
    la_wb = la.open_excel('data/test.xlsx')
else:
    print("skipping larray.Workbook test (xlwings not installed)")
    la_wb = None

# FIXME: the number of digits shown is 0 by default but should be 1,
#        because la_arr6 contains some ".5" values.
#        this is due to LArrayArrayAdapter.get_sample which returns 200 values
#        but not any from la_arr6.

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
# compare(la_int_2d, la_int_2d)
# compare(la_int_2d, la_int_2d + 1.0)
# compare(la_int_2d, la_int_2d + 1.0, names=['la_int_2d', 'la_int_2d + 1.0'])
# compare(np.random.normal(0, 1, size=(10, 2)), np.random.normal(0, 1, size=(10, 2)))

arr1 = la.ndtest((2, 3))
arr2 = la.ndtest((3, 4))
arr1bis = arr1.copy()
arr2bis = arr2.copy().drop('b3')
arr1bis['a1', 'b1'] = 42
arr2bis['a1', 'b1'] = 42
# arr2bis = arr2bis.set_labels({'b2': 'B2'})
# compare(arr2, arr2bis, align='outer')
# compare(arr2bis, arr2, align='outer')
# compare(arr2, arr2bis, align='inner')
# compare(arr2, arr2bis, align='left')
# compare(arr2, arr2bis, align='right')
# compare(arr2, arr2bis, align='exact')

s1 = la.Session(arr1=arr1, arr2=arr2)
s2 = la.Session(arr1=arr1bis, arr2=arr2bis)
del arr1, arr2, arr1bis, arr2bis
# compare(s1, s2, align="outer", rtol=10)

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
# compare(arr1, arr2, title='with nans on both sides (nans_equal=True)')
# compare(arr1, arr2, nans_equal=False, title='with nans on both sides (nans_equal=False)')
# arr1['M'] = arr1_m
# compare(arr1, arr2, nans_equal=False, title='with nans on right side')
#
# arr1 = la.ndtest((3, 3))
# arr2 = 2 * arr1
# arr3 = la.where(arr1 % 2 == 0, arr1, -arr1)
# compare(arr1, arr2, arr3, bg_gradient='blue-white-red', title='changed gradient')

# arr1 = la.ndtest(3).astype(np.float64)
# arr2 = arr1.copy()
# arr2['a0'] = 42
# arr3 = arr2.copy()
# arr3['a1'] = 4
# arr4 = arr1.copy()
# arr4['a1'] = la.inf
# arr5 = arr1.copy()
# arr5['a1'] = -la.inf
# arr6 = arr1.copy()
# arr6['a1'] = 1.00000000001
# compare(arr1, arr2)
# compare(arr2, arr1)
# compare(arr1, arr1)
# compare(arr1, arr3)
# compare(arr3, arr1)
# compare(arr1, arr4)
# compare(arr4, arr1)
# compare(arr1, arr5)
# compare(arr5, arr1)
# compare(arr5, arr5)
# compare(arr1, arr6)
# compare(arr6, arr1)

def test_compare_with_file_path():
    from larray_editor.api import compare

    sess1 = la.Session(arr4=la_int_2d, arr3=la_float_round_values,
                       data=np_arr2d)
    sess1.save('sess1.h5')
    sess2 = la.Session(arr4=la_int_2d + 1.0, arr3=la_float_round_values * 2.0,
                       data=np_arr2d * 1.05)
    # sess1.h5/data is nan because np arrays are not saved to H5
    # using a string path
    compare('sess1.h5', sess2)
    # using a Path object
    compare(Path('sess1.h5'), sess2)
    Path('sess1.h5').unlink()

# test_compare_with_file_path()


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
def make_test_df(size, offset=0):
    return pd.DataFrame({
        'name': la.sequence(size, initial=offset).apply(lambda i: f'name{i}').to_series(),
        'age': la.random.randint(0, 105, axes=size).to_series(),
        'male': (la.random.randint(0, 2, axes=size) == 1).to_series(),
        'height': la.random.normal(1.75, 0.07, axes=size).to_series()
    })

pd_df_mixed = make_test_df(100_000)
pd_df1 = la_int_2d.df
pd_df2 = la_float_4d_many_digits.df
pd_df3 = pd_df2.T
pd_df4 = pd_df2.unstack()
pd_df_str = pd_df2.astype(str)
pd_series = pd_df2.stack()

pd_df_big = la_big3d.df

if not Path('data/big.parquet').exists():
    print("Generating big.parquet test files (this may take a while)...",
          end=' ', flush=True)
    _big_no_idx = pd_df_big.reset_index()
    _big_no_idx.to_parquet('data/big.parquet')
    # Polars seems to have issues with Feather files written by Pandas
    # _big_no_idx.to_feather('data/big.feather')
    del _big_no_idx
    print("done.")

if not Path('data/big.h5').exists():
    print("Generating big.h5 test file...", end=' ', flush=True)
    la_big3d.to_hdf('data/big.h5', key='data')
    print("done.")

if not Path('data/big.csv').exists():
    print("Generating big.csv test file...", end=' ', flush=True)
    la_big3d.to_csv('data/big.csv')
    print("done.")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pyarrow_int_array = pa.array([2, 4, 5, 42])
    pyarrow_str_array = pa.array(["Hello", "from", "Arrow", "!"])
    pyarrow_table = pa.Table.from_arrays([pyarrow_int_array, pyarrow_str_array],
                                         names=["int_col", "str_col"])

    pyarrow_parquet_file = pq.ParquetFile('data/big.parquet')

    def gen_feather_file(fpath):
        print("Generating big.feather test file...", end=' ', flush=True)
        BATCH_SIZE = 10_000
        NUM_BATCHES = 10_000
        schema = pa.schema([
            pa.field('name', pa.string()),
            pa.field('age', pa.int32()),
            pa.field('male', pa.bool_()),
            pa.field('height', pa.float32()),
        ])
        with pa.OSFile(fpath, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                for batch_num in range(NUM_BATCHES):
                    batch_df = make_test_df(BATCH_SIZE,
                                            offset=batch_num * BATCH_SIZE)
                    batch = pa.RecordBatch.from_pandas(batch_df, schema=schema)
                    writer.write(batch)
        print("done.")

    if not Path('data/big.feather').exists():
        gen_feather_file('data/big.feather')
except ImportError:
    print("skipping pyarrow tests (not installed)")

try:
    import polars as pl

    pl_df1 = pl.from_pandas(pd_df1)
    pl_df2 = pl.from_pandas(pd_df2)
    # test with a datetime column and another column
    # the Arrow table has the same problem (e.g. pl_df3.to_arrow())
    pl_df3 = pl_df1.select(pl.from_epoch(pl.col('M')).alias('datetime_col'), 'M').limit(5)
    pl_df_big = pl.from_pandas(pd_df_big, include_index=True)
    pl_df_mixed = pl.from_pandas(pd_df_mixed, include_index=False)
    pl_lf_parquet = pl.scan_parquet('data/big.parquet')
    pl_lf_feather = pl.scan_ipc('data/big.feather')

    try:
        import narwhals as nw

        nw_df = nw.from_native(pl_df_mixed)
        nw_lf = nw.from_native(pl_lf_parquet)
    except ImportError:
        print("skipping narwhals tests (not installed)")

except ImportError:
    print("skipping polars tests (not installed)")


path_dir = Path('.')
path_py = Path('test_adapter.py')
path_csv = Path('data/big.csv')

# import cProfile as profile
# profile.runctx('edit(la.Session(arr2=arr2))', vars(), {},
#                'c:/tmp/edit.profile')
# import pstats
# pstats_stats = pstats.Stats('c:\\tmp\\edit.profile')

sqlite_con = sqlite3.connect(":memory:")
cur = sqlite_con.cursor()
cur.execute("create table lang (name, first_appeared)")
cur.executemany("insert into lang values (?, ?)", list_mixed_tuples)
cur.close()

try:
    import duckdb

    # in-memory duckdb database
    duckdb_con = duckdb.connect(":memory:")
    duckdb_con.execute("create table lang (name VARCHAR, first_appeared INTEGER)")
    duckdb_con.executemany("insert into lang values (?, ?)", list_mixed_tuples)
    duckdb_table = duckdb_con.table('lang')

    if not Path('data/test.duckdb').exists():
        print("Generating test.duckdb test file...", end=' ', flush=True)
        duckdb_con.execute("""
ATTACH 'data/test.duckdb' AS file_db;
COPY FROM DATABASE memory TO file_db;
DETACH file_db;""")
        duckdb_file_con = duckdb.connect('data/test.duckdb')
        duckdb_file_con.execute("CREATE TABLE big AS SELECT * FROM "
                                "read_parquet('data/big.parquet')")
        duckdb_file_con.close()
        print("done.")

except ImportError:
    print("skipping duckdb tests (not installed)")

zipf = zipfile.ZipFile('data/test.zip')

edit()
# debug()

# test_edit_after_matplotlib_show()
