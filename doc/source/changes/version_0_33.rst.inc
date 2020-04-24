.. py:currentmodule:: larray_editor

Syntax changes
^^^^^^^^^^^^^^

* renamed ``MappingEditor.old_method_name()`` to :py:obj:`MappingEditor.new_method_name()` (closes :editor_issue:`1`).

* renamed ``old_argument_name`` argument of :py:obj:`MappingEditor.method_name()` to ``new_argument_name``.


Backward incompatible changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* dropped Python 2 support (closes :editor_issue:`132`).


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

* workaround incompatibility with Python3.8 on Windows (closes :editor_issue:`208`).

* workaround incompatibility between two of our dependencies versions preventing the editor to even start
  (closes :editor_issue:`209`).
