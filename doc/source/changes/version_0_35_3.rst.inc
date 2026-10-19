.. py:currentmodule:: larray_editor

New features
^^^^^^^^^^^^

* Axis and Group objects (and Pandas Series) stored in HDF (.h5) files can now be
  displayed (closes :editor_issue:`326`).

* .json files are now displayed as text.


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* when hovering over a plot made from the editor interface, the X coordinate shown in the
  plot window is now the corresponding label(s) (closes :editor_issue:`336`).

* when zooming on such plots, X axis tick labels which were hidden for lack of space can now
  be shown on the zoomed plot.

* pasting values in more than 100,000 cells (e.g. a single value in a whole column of a
  large array) now asks for confirmation first, because it can take a while.


Fixes
^^^^^

* fixed spurious "received too many columns" warnings displayed in the console, notably
  when using :py:obj:`compare()` (closes :editor_issue:`333`).

* fixed opening files whose path contains spaces via the Windows file association
  (closes :editor_issue:`332`).

* sorted arrays are read-only for now: modifying one of their values modified another
  value instead.

* fixed comparing arrays with equal anonymous axes (axes without name). An error was
  displayed since version 0.35. Comparing arrays whose anonymous axes differ still fails.

* read-only editor windows, such as those opened by :py:obj:`view()`, are read-only again
  (values could be modified since version 0.35).

* fixed text and CSV files larger than 64 KB: from some line on, lines could be displayed
  in the wrong row (repeated or shifted by one row) and the files seemed to contain more
  lines than they do.

* the displayed object is now refreshed after any console command which does not display
  another object, because the command (e.g. a function call) could have modified it
  (closes :editor_issue:`335`).

* fixed modifying string values with numpy >= 2.

* fixed modifying values of arrays with the object dtype (e.g. arrays mixing numbers and
  strings).

* modifying values marks the session as modified again (with a * in the window title),
  as before version 0.35.

* pasting data which extends past the last row or column of the array now pastes the
  part which fits and displays a warning (an error was displayed and nothing was pasted).

* pasting text whose lines do not all contain the same number of values now displays a
  clear error message.

* fixed pasting values in large selections (e.g. a whole column of an array with many
  rows).

* background colors are now updated when values are modified, including by undo/redo
  (closes :editor_issue:`159`).

* fixed filtering date and time columns of Polars DataFrames and LazyFrames (e.g. Parquet
  files) (closes :editor_issue:`292`).

* fixed displaying DataFrames stored in HDF (.h5) files which contain only string columns, or
  which contain string columns and were written using Pandas >= 3.

* fixed displaying arrays and DataFrames stored in HDF (.h5) files when their labels
  are of mixed types (e.g. both strings and numbers) (closes :editor_issue:`330`).

* the scrollbars of filtered Pandas DataFrames now take the filter into account (it was
  possible to scroll past the filtered rows, to empty rows).

* the results of expressions are now displayed read-only. Modifying their values failed
  (closes :editor_issue:`155`).

* fixed Polars DataFrames and Series being displayed as generic collections when passed directly
  to :py:obj:`edit()` or :py:obj:`view()` (e.g. ``edit(df)``).

* datetimes in Pandas Series are now displayed like in Pandas DataFrames (e.g.
  2024-01-01 00:00:00 instead of 2024-01-01T00:00:00.000000).

* integers with missing values (in Pandas, Polars, narwhals and pyarrow objects) are now
  displayed as integers, with None for missing values. They were converted to floats, which
  showed as such (e.g. 1.0) next to columns of other types.

* fixed displaying time durations (timedelta), e.g. columns of DataFrames.

* fixed row labels and axis names sometimes being partially hidden after switching from
  one array to another (their area kept the width it had for the previous array).

* all files opened by the viewer are now closed when another object is displayed.
  Previously, only Excel and HDF files were closed. Other types were kept open, which prevented
  renaming or deleting them on Windows while the viewer was open.

* fixed the selection when dragging the mouse on row or column labels and going back
  (closes :editor_issue:`107`).

* LArray Groups are now displayed as their labels. In particular, a group with a single string
  label was displayed as a sequence of characters (closes :editor_issue:`314`).

* fixed variables defined in a multi-line console command not appearing in the variable
  list when the last line of that command modified an existing variable
  (closes :editor_issue:`334`).

* filtering Polars objects on missing values (null) now works (it selected nothing).

* values displayed in scientific notation now show as many significant digits as needed
  (and fit). For example, 150000000 was displayed as 2e+08 instead of 1.5e+08.

* pasting with an empty clipboard no longer displays an error (it just does nothing instead).

* fixed displaying empty HDF (.h5) files (closes :editor_issue:`325`) and Feather files
  without any row.

* fixed displaying DataFrames (and other tables) without any column.

* fixed displaying empty CSV files, which made the editor hang.

* the labels of arrays with an empty axis are now displayed (the labels of all axes were
  missing).

* fixed row labels of arrays and tables without any column (e.g. a 2D array whose last axis
  is empty) being only partially displayed (their column was not resized to fit them). The
  names of their axes are now displayed too.

* fixed a spurious "received more rows than expected" warning when displaying arrays and
  tables without any row.

* fixed modifying values when the Python console is disabled or unavailable (the editor
  crashed).

* fixed displaying DuckDB connections and relations with DuckDB 1.4 or later.

* fixed displaying pyarrow Tables. They could not be displayed at all unless the pyarrow.parquet
  module was imported, and even then, not the first time one was passed directly to
  :py:obj:`edit()` or :py:obj:`view()`.

* subclasses of supported types (e.g. of numpy arrays) are now displayed like their parent class
  instead of as generic collections.

* fixed displaying narwhals DataFrames with columns of different types and filtering them.

* fixed filtering narwhals Series.

* fixed a performance warning displayed when filtering Polars LazyFrames (e.g. Parquet
  files).

* complex numbers are now displayed with their imaginary part.

* fixed :py:obj:`edit()` with ``REOPEN_LAST_FILE`` when there were recently opened files.

* fixed using the keyboard in the sort menu of the column labels of 2D arrays, lists or
  dictionaries (menus without filter).

* fixed autocompletion in the SQL console when the query is a single word ending with an
  underscore (e.g. ``my_``).

* the history of the SQL console is now limited to the last 1000 queries, as intended. When
  the history was longer than that, the most recent queries were skipped when navigating it
  with the up/down arrow keys.

* fixed pressing the Delete key when the list of variables is empty.

* fixed displaying IODE scalars files with the .scl extension.

* the list of sheets of Excel files no longer offers to sort them, which did not work.

* error messages containing text which looks like HTML tags (e.g. "<class 'int'>") are now
  displayed fully in the error dialog (that text was missing).

* fixed copying, pasting and plotting a selection extended with Shift+click: only the clicked
  cell was used (since version 0.35).

* filtering or sorting no longer keeps an invisible selection: nothing was displayed as
  selected afterwards, but copying and pasting still used the previously selected cells.
  Since version 0.35.

* fixed text and CSV files larger than 4 MB with Windows line endings: in some cases, from some
  line on, two lines were displayed as one and all the following lines one row too high.

* fixed the encoding of some UTF-8 text and CSV files larger than 64 KB being misdetected, which
  garbled their accented characters.

* fixed :py:obj:`edit()` and :py:obj:`view()` displaying Pandas DataFrames and Series passed
  directly (e.g. ``view(df)``) as a list of variables (one per column or value) instead of
  displaying them.

* fixed the ``root_path`` argument of :py:obj:`run_editor_on_exception()` when it is a relative path
  or uses forward slashes on Windows.
