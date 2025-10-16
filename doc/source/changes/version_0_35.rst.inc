.. py:currentmodule:: larray_editor

New features
^^^^^^^^^^^^

* allow displaying *many* more different kinds of objects, and not only arrays
  from larray. We only support displaying (not editing) all those types so far:
  * Python builtin objects:
    - tuple, list (sequences), dict (mappings), dict views, memoryview, array
    - text and binary files
  * Python stdlib objects:
    - pathlib.Path
      * if the path points to a directory, it will display the content of the
        directory
      * if the path points to a file, it will try to display it, if we
        implemented support for that file type (see below for the list
        of supported types).
    - sqlite3.Connection (and their tables)
    - pstats.Stats (results of Python's profiler)
    - zipfile.ZipFile and zipfile.Path
  * other objects from LArray:
    - Axis
    - Excel Workbook (what you get from larray.open_excel()), Sheets and Range
  * Pandas:
    - DataFrame
    - Series
  * Numpy:
    - ndarray
  * pyarrow:
    - Array
    - Table
    - RecordBatchFileReader (reader object for feather files)
    - ParquetFile
  * Polars:
    - DataFrame
    - LazyFrame
  * Narwhals:
    - DataFrame
    - LazyFrame
  * IODE:
    - Variables
    - Table
    - Comments
    - Identities
    - Lists
    - Tables
    - Scalars
    - Equations
  * PyTables:
    - Table
  * IBIS:
    - Table
  * DuckDB:
    - DuckDBPyConnection
    - DuckDBPyRelation (what you receive from any query)

  File types (extensions) currently supported:
    - Iode files: .ac, .ae, .ai, .al, .as, .at, .av, .cmt, .eqs, .idt, .lst, .scl, .tbl, .var
    - Text files: .bat, .cfg, .md, .py, .rst, .sh, .toml, .txt, .yaml, .yml
    - Parquet files: .parquet
    - CSV files: .csv
    - Excel files: .xls, .xlsx
    - Zip files: .zip
    - DuckDB files: .ddb, .duckdb
    - SAS files: .sas7bdat

* allow comparing arrays/sessions with different axes in :py:obj:`compare()`.
  The function gained ``align`` and ``fill_value`` arguments and the interface has a new
  combobox to change the alignment method during the comparison:
      - outer: will use a label if it is in any array (ordered like the first array).
               This is the default as it results in no information loss.
      - inner: will use a label if it is in all arrays (ordered like the first array).
      - left: will use the first array axis labels.
      - right: will use the last array axis labels.
      - exact: raise an error when axes are not equal.
  Closes :editor_issue:`214` and :editor_issue:`251`.

* double-clicking on an array name in the list will open it in a new window
  (closes :editor_issue:`143`).


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* made the editor interruptible by an outside program (i.e. made PyCharm stop & restart buttons work directly
  instead of only when the editor receives the focus again). Closes :editor_issue:`257`.

* resize axes and vertical label columns automatically

* string values are left aligned instead of right aligned

* when comparing sessions via :py:obj:`compare()`, the color of arrays in the list is now updated depending
  on the tolerance. To reflect that the tolerance widget moved to the top of the interface.
  Closes :editor_issue:`201`.

* :py:obj:`compare()` max difference is colored red when the difference is not 0

* typing the name of a variable holding a matplotlib figure (or axes) in the console shows it
  (previously, only expressions were displayed and *not* simple variables).
  For example: ::

     >>> arr.plot()

  already displayed a plot, but: ::

     >>> ax = arr.plot()
     >>> ax

  did not.

* display matplotlib Figure objects in addition to Axes objects


Fixes
^^^^^

* fixed :py:obj:`compare()` colors when the only difference is nans on either
  side.

* fixed single column plot in viewer when ticks are not strings