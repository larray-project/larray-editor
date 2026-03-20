.. py:currentmodule:: larray_editor

New features
^^^^^^^^^^^^

* added explicit support for Python 3.14.


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Files passed as command-line arguments to the editor are now displayed via
  the "File Explorer" mechanism instead of being loaded as a Session
  (closes :editor_issue:`303`).
  In practice, this means we can support many more file formats (.feather,
  .parquet, .dta, .sas7bdat, ...) and it is possible to open files which do not
  fit entirely in memory and much faster to open other large files.

  On the other hand, we loose the ability to *directly* manipulate the data as
  arrays, unless the user loads the file explicitly. Assuming the format is
  supported by Session (.h5, larray-formatted .xlsx or directory of .csv), this
  can be done either via the File->Open Data menu or via this command in the
  console:

    Session(name_of_the_variable_with_the_filepath).to_globals()


Fixes
^^^^^

* fixed displaying DataFrames with Pandas >= 3 (closes :editor_issue:`308`).

* fixed some objects not being updated immediately when updated. This includes
  updates to Pandas DataFrames done via `df.loc[key] = value` and
  `df.iloc[position] = value` (closes :editor_issue:`310`).

* fixed File Explorer on Python 3.9 (closes :editor_issue:`300`).

* fixed displaying "Collection" objects which raise an error on computing
  their length and/or iterating them, such as scalar Groups (closes
  :editor_issue:`313`).

* fixed the mechanism writing warning/error messages happening during the
  editor initialization. The errors are now correctly written in the user
  TEMP directory / larray-editor-stderr.log