.. py:currentmodule:: larray_editor

New features
^^^^^^^^^^^^

* added explicit support for Python 3.14.


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* improved something.


Fixes
^^^^^

* fixed displaying DataFrames with Pandas >= 3 (closes :editor_issue:`308`).

* fixed some objects not being updated immediately when updated. This includes
  updates to Pandas DataFrames done via `df.loc[key] = value` and
  `df.iloc[position] = value` (closes :editor_issue:`310`).