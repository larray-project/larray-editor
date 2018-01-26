from __future__ import absolute_import, print_function

import logging

from qtpy.QtWidgets import QUndoCommand
from larray_editor.utils import logger


class ArrayValueChange(object):
    """
    Class representing the change of one value of an array.

    Parameters
    ----------
    key: list/tuple of str
        Key associated with the value
    new_value: scalar
        New value
    old_value: scalar
        Previous value
    """
    def __init__(self, key, old_value, new_value):
        self.key = key
        self.old_value = old_value
        self.new_value = new_value


# XXX: we need to handle the case of several changes at once because the method paste()
#      of ArrayEditorWidget can be used on objects not handling MultiIndex axes (LArray, Numpy).
class EditArrayCommand(QUndoCommand):
    """
    Class representing the change of one or several value(s) of an array.

    Parameters
    ----------
    editor: MappingEditor
        Instance of MappingEditor
    changes: (list of) instance(s) of ArrayValueChange
        List of changes
    """

    def __init__(self, editor, array_name, changes):
        QUndoCommand.__init__(self)
        self.editor = editor
        self.array_name = array_name
        if not isinstance(changes, (list, tuple)):
            changes = (changes,)
        self.changes = changes

        if len(changes) == 1:
            text_command = "Editing Cell {} of {}".format(changes[0].key, array_name)
        else:
            text_command = "Pasting {} Cells in {}".format(len(changes), array_name)
        self.setText(text_command)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Edit command pushed: {}".format(text_command))

    def undo(self):
        for change in self.changes:
            command = "{}[{}] = {}".format(self.array_name, change.key, change.old_value)
            self.editor.kernel.shell.run_cell(command)
        self.editor.arraywidget.model_data.reset()

    def redo(self):
        for change in self.changes:
            command = "{}[{}] = {}".format(self.array_name, change.key, change.new_value)
            self.editor.kernel.shell.run_cell(command)
        self.editor.arraywidget.model_data.reset()
