from __future__ import absolute_import, print_function

from qtpy.QtWidgets import QUndoCommand


class EditArrayCommand(QUndoCommand):
    """
    Class representing the change of one value of an array.

    Parameters
    ----------
    editor: MappingEditor
        Instance of MappingEditor
    key: list of str
        Key associated with the value
    new_value: scalar
        New value
    old_value: scalar
        Previous value
    """

    def __init__(self, editor, array_name, key, old_value, new_value):
        QUndoCommand.__init__(self)
        self.editor = editor
        self.array_name = array_name
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        command = "{}[{}] = {}".format(self.array_name, self.key, self.new_value)
        self.setText(command)

    def undo(self):
        command = "{}[{}] = {}".format(self.array_name, self.key, self.old_value)
        print("undo {}".format(command))
        # move next 2 lines to MappingEditor?
        self.editor.kernel.shell.run_cell(command)
        self.editor.arraywidget.model_data.reset()

    def redo(self):
        command = "{}[{}] = {}".format(self.array_name, self.key, self.new_value)
        print("redo {}".format(command))
        # move next 2 lines to MappingEditor?
        self.editor.kernel.shell.run_cell(command)
        self.editor.arraywidget.model_data.reset()
