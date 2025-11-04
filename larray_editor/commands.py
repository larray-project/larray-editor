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
    # FIXME: key is a tuple of indices
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

        text_command = self.get_description(target, changes)
        self.setText(text_command)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Edit command pushed: {text_command}")

    def undo(self):
        for change in self.changes:
            self.apply_change(change.key, change.old_value)
        # FIXME: a full reset is bad, see comment below
        self.editor.arraywidget.model_data.reset()

    def redo(self):
        for change in self.changes:
            self.apply_change(change.key, change.new_value)
        # FIXME: a full reset is both wasteful, and causes hidden scrollbars
        #        to jump back to 0 after each cell change, which is very
        #        annoying. We have an awful workaround for this in
        #        ArrayDelegate.setModelData but the issue should still be fixed
        #        properly
        self.editor.arraywidget.model_data.reset()

    def get_description(self, target, changes):
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

    def apply_change(self, key, new_value):
        # FIXME: we should pass via the adapter to have something generic
        self.editor.ipython_kernel.shell.run_cell(f"{self.target}.i[{key}] = {new_value}")


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

    def apply_change(self, key, new_value):
        # FIXME: we should pass via the adapter to have something generic
        self.target.i[key] = new_value
