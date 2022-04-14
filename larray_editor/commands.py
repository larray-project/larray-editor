import logging

try:
    from qtpy.QtWidgets import QUndoCommand
except ImportError:
    # PySide6 provides QUndoCommand in QtGui
    # qtpy has been fixed (see https://github.com/spyder-ide/qtpy/pull/366) but the version available via conda as of
    # 20/09/2022 (2.2) does not include the fix yet
    from qtpy.QtGui import QUndoCommand

from larray_editor.utils import logger


class ArrayValueChange:
    """
    Class representing the change of one value of an array.

    Parameters
    ----------
    key: list/tuple of str
        Key associated with the value
    old_value: scalar
        Previous value
    new_value: scalar
        New value
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
    target : object
        target array to edit. Can be given under any form.
    changes: (list of) instance(s) of ArrayValueChange
        List of changes
    """

    def __init__(self, editor, target, changes):
        QUndoCommand.__init__(self)
        self.editor = editor
        self.target = target
        assert isinstance(changes, list)
        self.changes = changes

        text_command = self.get_description(target, changes)
        self.setText(text_command)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Edit command pushed: {text_command}")

    def undo(self):
        for change in self.changes:
            self.apply_change(change.key, change.old_value)
        self.editor.arraywidget.model_data.reset()

    def redo(self):
        for change in self.changes:
            self.apply_change(change.key, change.new_value)
        self.editor.arraywidget.model_data.reset()

    def get_description(self, target, changes):
        raise NotImplementedError()

    def apply_change(self, key, new_value):
        raise NotImplementedError()


class EditSessionArrayCommand(EditArrayCommand):
    """
    Class representing the change of one or several value(s) of an array.

    Parameters
    ----------
    editor: MappingEditor
        Instance of MappingEditor
    target : str
        name of array to edit
    changes: (list of) instance(s) of ArrayValueChange
        List of changes
    """
    def get_description(self, target, changes):
        if len(changes) == 1:
            return f"Editing Cell {changes[0].key} of {target}"
        else:
            return f"Pasting {len(changes)} Cells in {target}"

    def apply_change(self, key, new_value):
        self.editor.kernel.shell.run_cell(f"{self.target}[{key}] = {new_value}")


class EditCurrentArrayCommand(EditArrayCommand):
    """
    Class representing the change of one or several value(s) of the current array.

    Parameters
    ----------
    editor : ArrayEditor
        Instance of ArrayEditor
    target : Array
        array to edit
    changes : (list of) instance(s) of ArrayValueChange
        List of changes
    """
    def get_description(self, target, changes):
        if len(changes) == 1:
            return f"Editing Cell {changes[0].key}"
        else:
            return f"Pasting {len(changes)} Cells"

    def apply_change(self, key, new_value):
        self.target[key] = new_value
