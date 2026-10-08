.. py:currentmodule:: larray_editor

Syntax changes
^^^^^^^^^^^^^^

* renamed ``MappingEditor.old_method_name()`` to :py:obj:`MappingEditor.new_method_name()` (closes :editor_issue:`1`).

* renamed ``old_argument_name`` argument of :py:obj:`MappingEditor.method_name()` to ``new_argument_name``.


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* other backward incompatible changes


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


Fixes
^^^^^

* fixed opening files whose path contains spaces via the Windows file association
  (closes :editor_issue:`332`).

* read-only editor windows, such as those opened by :py:obj:`view()`, are read-only again
  (values could be modified since version 0.35).

* fixed spurious "received too many columns" warnings displayed in the console, notably
  when using :py:obj:`compare()` (closes :editor_issue:`333`).

* the displayed object is now refreshed after any console command which does not display
  another object, because the command (e.g. a function call) could have modified it
  (closes :editor_issue:`335`).

* fixed modifying string values with numpy >= 2 (an error was displayed).

* fixed modifying values when the Python console is disabled or unavailable (the editor
  crashed).

* background colors are now updated when values are modified, including by undo/redo
  (closes :editor_issue:`159`).

* the results of expressions are now displayed read-only. Modifying their values failed
  (closes :editor_issue:`155`).

* fixed filtering date and time columns of Polars DataFrames and LazyFrames (e.g. Parquet
  files) (closes :editor_issue:`292`).

* the scrollbars of filtered Pandas DataFrames now take the filter into account (it was
  possible to scroll past the filtered rows, to empty rows).

* fixed displaying DuckDB connections and relations with DuckDB 1.4 or later.

* fixed Polars DataFrames and Series being displayed as generic collections when passed directly
  to :py:obj:`edit()` or :py:obj:`view()` (e.g. ``edit(df)``).

* fixed displaying pyarrow Tables. They could not be displayed at all unless the pyarrow.parquet
  module was imported, and even then, not the first time one was passed directly to
  :py:obj:`edit()` or :py:obj:`view()`.

* subclasses of supported types (e.g. of numpy arrays) are now displayed like their parent class
  instead of as generic collections.

* fixed displaying DataFrames stored in HDF (.h5) files which contain only string columns, or
  which contain string columns and were written using Pandas >= 3.

* fixed displaying arrays and DataFrames stored in HDF (.h5) files when their labels
  are of mixed types (e.g. both strings and numbers) (closes :editor_issue:`330`).

* fixed variables defined in a multi-line console command not appearing in the variable
  list when the last line of that command modified an existing variable
  (closes :editor_issue:`334`).

* fixed displaying time durations (timedelta), e.g. columns of DataFrames.

* fixed displaying empty HDF (.h5) files (closes :editor_issue:`325`) and Feather files
  without any row.

* fixed displaying DataFrames (and other tables) without any column.

* fixed displaying empty CSV files, which made the editor hang.

* files opened to display them (text, CSV, zip, DuckDB, ... files) are now closed when another
  object is displayed. They were kept open, which prevented renaming or deleting them on
  Windows while the editor was open.

* fixed the selection when dragging the mouse on row or column labels and going back
  (closes :editor_issue:`107`).

* groups are now displayed as their labels. In particular, a group with a single string
  label was displayed as a sequence of characters (closes :editor_issue:`314`).

* the labels of arrays with an empty axis are now displayed (the labels of all axes were
  missing).

* filtering Polars objects on missing values (null) now works (it selected nothing).

* fixed displaying narwhals DataFrames with columns of different types and filtering them.

* fixed filtering narwhals Series.

* fixed a performance warning displayed when filtering Polars LazyFrames (e.g. Parquet
  files).
