import sqlite3

from larray_editor.arrayadapter import SQLiteTable, get_adapter
from larray_editor.tests.adapters.common import adapter_values


def test_sqlite_names_needing_quotes():
    con = sqlite3.connect(':memory:')
    # a space in the table name and column names with a space and a reserved word
    con.execute('CREATE TABLE "my table" ("my col" INTEGER, "order" INTEGER)')
    con.executemany('INSERT INTO "my table" VALUES (?, ?)', [(2, 1), (1, 2)])
    adapter = get_adapter(SQLiteTable(con, 'my table'))
    adapter.sort_hlabel(0, 0, True)
    rows = adapter_values(adapter)
    con.close()
    assert rows == [['', 'my col', 'order'],
                    [0, 1, 2],
                    [1, 2, 1]]
