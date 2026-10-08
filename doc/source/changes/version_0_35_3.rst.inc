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

* fixed spurious "received too many columns" warnings displayed in the console, notably
  when using :py:obj:`compare()` (closes :editor_issue:`333`).

* fixed displaying arrays and DataFrames stored in HDF (.h5) files when their labels
  are of mixed types (e.g. both strings and numbers) (closes :editor_issue:`330`).

* fixed variables defined in a multi-line console command not appearing in the variable
  list when the last line of that command modified an existing variable
  (closes :editor_issue:`334`).
