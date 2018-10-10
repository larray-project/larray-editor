.. py:currentmodule:: larray_editor

New features
^^^^^^^^^^^^

* added :py:obj:`debug()` function which opens an editor window with an extra widget to navigate back in the call
  stack (the chain of functions called to reach the current line of code).


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* added keyword arguments ``rtol``, ``atol`` and ``nans_equal`` to the :py:obj:`compare()` function
  (closes :editor_issue:`172`).

* :py:obj:`run_editor_on_exception()` now uses :py:obj:`debug()` so that one can inspect what the state was in all
  functions traversed to reach the code which triggered the exception.


Fixes
^^^^^

* fixed something (closes :editor_issue:`1`).
