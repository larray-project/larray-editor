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

  .. note::

     - It works for foo bar !
     - It does not work for foo baz !


Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* made the editor interruptible by an outside program (i.e. made PyCharm stop & restart buttons work directly
  instead of only when the editor receives the focus again). Closes :editor_issue:`257`.

* when comparing sessions via :py:obj:`compare()`, the color of arrays in the list is now updated depending
  on the tolerance. To reflect that the tolerance widget moved to the top of the interface.
  Closes :editor_issue:`201`.

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

* fixed :py:obj:`compare()` colors when the only difference is nans on either side.

* fixed something (closes :editor_issue:`1`).
