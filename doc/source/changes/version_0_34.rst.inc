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

* added support for Python 3.10.

* the initial column width is now set depending on the content and resized automatically when changing the number of
  digits (closes :editor_issue:`145`).

* added a feature (see the :ref:`miscellaneous section <misc_editor>` for details).

* added another feature in the editor (closes :editor_issue:`1`).

  .. note::

     - It works for foo bar !
     - It does not work for foo baz !


.. _misc_editor:

Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Plot windows title now include the expression used to make the plot (the name of the array
  in most cases) (closes :editor_issue:`233`).

* when displaying an expression (computed array), the window title includes the actual expression
  instead of using '<expr>'.

* When the editor displays currently defined variables (via `debug()` `edit()` or
  `view()` without argument within user code or via an exception when run_editor_on_exception is active),
  LArray functions are not made available systematically in the console anymore (what is available in the console is
  really what was available in the users code). This closes :editor_issue:`199`.


Fixes
^^^^^

* fixed `run_editor_on_exception` so that the larray editor is not opened when trying to stop a program (via Ctrl-C or
  the IDE stop button). Closes :editor_issue:`231`.

* fixed the larray editor window not closing when trying to stop a program (via `Ctrl-C` or the IDE stop button) with
  such a window open (closes :editor_issue:`231`). With PyCharm for example, to actually close the program, the user
  had to press the IDE stop button a second time (which killed the program instead of stopping it cleanly).
