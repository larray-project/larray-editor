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

* added a feature (see the :ref:`miscellaneous section <misc_editor>` for details).

* added another feature in the editor (closes :editor_issue:`1`).

  .. note::

     - It works for foo bar !
     - It does not work for foo baz !


.. _misc_editor:

Miscellaneous improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^

* improved something.


Fixes
^^^^^

* fixed spurious warning in the console when an expression results in an empty sequence (array, list, tuple).
