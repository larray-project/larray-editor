import logging

try:
    from qtpy.QtWidgets import QUndoCommand
except ImportError:
    # PySide6 provides QUndoCommand in QtGui
    # qtpy has been fixed (see https://github.com/spyder-ide/qtpy/pull/366) but the version available via conda as of
    # 20/09/2022 (2.2) does not include the fix yet
    from qtpy.QtGui import QUndoCommand

from larray_editor.utils import logger


class CellValueChange:
    """
    Class representing the change of one value of an array.

    Parameters
    ----------
    key: tuple of int
        Positional indices of the value
    new_value: scalar
        New value

    Attributes
    ----------
    old_value: scalar
        Previous value. Recorded by the command when it first applies the change.
    """
    def __init__(self, key, new_value):
        self.key = key
        self.new_value = new_value
        self.old_value = None


# XXX: we need to handle the case of several changes at once because the method paste()
#      of ArrayEditorWidget can be used on objects not handling MultiIndex axes (LArray, Numpy).
class EditObjectCommand(QUndoCommand):
    """
    Class representing the change of one or several value(s) of an array.

    Parameters
    ----------
    editor: AbstractEditorWindow
        Instance of AbstractEditorWindow
    target : object
        target object to edit. Can be given under any form.
    changes: list of CellValueChange
        List of changes
    """

    def __init__(self, editor, target, changes):
        QUndoCommand.__init__(self)
        self.editor = editor
        self.target = target
        assert isinstance(changes, list)
        self.changes = changes
        self.old_values_recorded = False

    def undo(self):
        for change in self.changes:
            self.apply_change(change.key, change.old_value)
        self.refresh_data()

    def redo(self):
        if not self.old_values_recorded:
            self.record_old_values()
        for change in self.changes:
            self.apply_change(change.key, change.new_value)
        self.refresh_data()

    def record_old_values(self):
        # The undo stack is linear, so the values are the same each time a command is redone
        # and they only need to be recorded the first time. Recording them here (instead of
        # when creating the changes) avoids fetching the old values of the whole selection.
        for change in self.changes:
            change.old_value = self.get_value(change.key)
        self.changes = [change for change in self.changes
                        if change.new_value != change.old_value]
        self.old_values_recorded = True
        # an obsolete command is discarded by QUndoStack.push
        self.setObsolete(not self.changes)
        text_command = self.get_description(self.target, self.changes)
        self.setText(text_command)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Edit command pushed: {text_command}")

    def refresh_data(self):
        self.editor.array_widget.refresh_data()

    def get_description(self, target, changes):
        raise NotImplementedError()

    def get_value(self, key):
        raise NotImplementedError()

    def apply_change(self, key, new_value):
        raise NotImplementedError()


class EditSessionArrayCommand(EditObjectCommand):
    """
    Class representing the change of one or several value(s) of an array.

    Parameters
    ----------
    editor: MappingEditor
        Instance of MappingEditor
    target : str
        name of array to edit
    changes: (list of) instance(s) of CellValueChange
        List of changes
    """
    def get_description(self, target: str, changes: list[CellValueChange]):
        if len(changes) == 1:
            return f"Editing Cell {changes[0].key} of {target}"
        else:
            return f"Pasting {len(changes)} Cells in {target}"

    def get_value(self, key):
        return self.editor.data[self.target].i[key]

    def apply_change(self, key, new_value):
        # FIXME: we should pass via the adapter to have something generic
        self.editor.data[self.target].i[key] = new_value


class EditCurrentArrayCommand(EditObjectCommand):
    """
    Class representing the change of one or several value(s) of the current array.

    Parameters
    ----------
    editor : ArrayEditor
        Instance of ArrayEditor
    target : Array
        array to edit
    changes : (list of) ArrayValueChange
        List of changes
    """
    def get_description(self, target, changes):
        if len(changes) == 1:
            return f"Editing Cell {changes[0].key}"
        else:
            return f"Pasting {len(changes)} Cells"

    def get_value(self, key):
        return self.target.i[key]

    def apply_change(self, key, new_value):
        # FIXME: we should pass via the adapter to have something generic
        self.target.i[key] = new_value
