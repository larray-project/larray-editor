import sys
import types

import numpy as np
import pandas as pd

from larray_editor.arrayadapter import (AbstractAdapter, get_adapter, get_adapter_creator,
                                        register_adapter, update_registered_adapter_types,
                                        REGISTERED_ADAPTERS, REGISTERED_ADAPTERS_USING_STRINGS)
from larray_editor.tests.adapters.common import adapter_values


def test_adapter_registered_using_string():
    # Mimics libraries like duckdb (whose types are defined in a module with
    # another name) and polars (whose DataFrame is also a Collection)
    class FakeTable:
        __module__ = '_fake_lib_impl'

        def __len__(self):
            return 0

        def __iter__(self):
            return iter([])

        def __contains__(self, item):
            return False

    class FakeTableAdapter(AbstractAdapter):
        pass

    fake_lib = types.ModuleType('fake_lib')
    fake_lib.Table = FakeTable
    sys.modules['fake_lib'] = fake_lib
    try:
        # a type in a submodule which is not loaded must not prevent finding the others
        register_adapter('fake_lib.not_loaded.Other', AbstractAdapter)
        register_adapter('fake_lib.Table', FakeTableAdapter)
        table = FakeTable()
        assert get_adapter_creator(table) is FakeTableAdapter
    finally:
        del sys.modules['fake_lib']
        REGISTERED_ADAPTERS_USING_STRINGS.pop('fake_lib', None)
        REGISTERED_ADAPTERS.pop(FakeTable, None)
        update_registered_adapter_types()


def test_adapter_for_subclass():
    class ArraySubclass(np.ndarray):
        pass

    array_adapter_creator = get_adapter_creator(np.zeros(2))
    subclass_array = np.zeros(2).view(ArraySubclass)
    # the adapter of the parent class must be used, not the one of an abstract
    # class (Collection) the subclass is a "virtual" subclass of
    assert get_adapter_creator(subclass_array) is array_adapter_creator


def test_no_column():
    df_without_column = pd.DataFrame(index=[1, 2])
    assert adapter_values(df_without_column) == [[None],
                                                 [1],
                                                 [2]]


def test_close_opened_file(tmp_path):
    fpath = tmp_path / 'test.txt'
    fpath.write_text('a')

    # files the adapter itself opens must be closed
    adapter = get_adapter(fpath)
    adapter.close()
    assert adapter.data.closed

    # files given (already opened) to the adapter must be left open
    with open(fpath) as f:
        get_adapter(f).close()
        assert not f.closed


def test_vlabels_without_row():
    adapter = get_adapter(np.zeros((0, 3)))
    vlabels = adapter.get_vlabels(0, 0)['values']
    # there must not be a row (without any label)
    assert vlabels == []
