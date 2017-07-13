import os
import re
import matplotlib
import numpy as np

from larray import LArray, Session, zeros
from larray_editor.utils import PYQT5, _, create_action, show_figure, ima
from larray_editor.arraywidget import ArrayEditorWidget

from qtpy.QtCore import Qt, QSettings, QUrl, Slot
from qtpy.QtGui import QDesktopServices, QKeySequence
from qtpy.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QSplitter, QFileDialog,
                            QPushButton, QDialogButtonBox, QAction, QShortcut, QHBoxLayout, QGridLayout,
                            QLineEdit, QMessageBox, QDialog)

try:
    from qtconsole.rich_jupyter_widget import RichJupyterWidget
    from qtconsole.inprocess import QtInProcessKernelManager
    from IPython import get_ipython

    ipython_instance = get_ipython()

    # Having several instances of IPython of different types in the same
    # process are not supported. We use
    # ipykernel.inprocess.ipkernel.InProcessInteractiveShell
    # and qtconsole and notebook use
    # ipykernel.zmqshell.ZMQInteractiveShell, so this cannot work.
    # For now, we simply fallback to not using IPython if we are run
    # from IPython (whether qtconsole or notebook). The correct solution is
    # probably to run the IPython console in a different process but I do not
    # know what would be the consequences. I fear it could be slow to transfer
    # the session data to the other process.
    if ipython_instance is None:
        qtconsole_available = True
    else:
        qtconsole_available = False
except ImportError:
    qtconsole_available = False

REOPEN_LAST_FILE = object()


assignment_pattern = re.compile('[^\[\]]+[^=]=[^=].+')
setitem_pattern = re.compile('(.+)\[.+\][^=]=[^=].+')
history_vars_pattern = re.compile('_i?\d+')
# XXX: add all scalars except strings (from numpy or plain Python)?
# (long) strings are not handled correctly so should NOT be in this list
# tuple, list
DISPLAY_IN_GRID = (LArray, np.ndarray)


class MappingEditor(QMainWindow):
    """Session Editor Dialog"""

    MAX_RECENT_FILES = 10

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        # to handle recently opened files
        settings = QSettings()
        # XXX: use recent_file_list?
        if settings.value("recentFileList") is None:
            settings.setValue("recentFileList", [])
        self.recent_file_actions = [QAction(self) for _ in range(self.MAX_RECENT_FILES)]
        self.current_file = None
        self.current_array = None
        self.current_array_name = None

        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.data = None
        self.arraywidget = None
        self._listwidget = None
        self.eval_box = None
        self.expressions = {}
        self.kernel = None
        self._unsaved_modifications = False

        self.setup_menu_bar()

    def setup_and_check(self, data, title='', readonly=False, minvalue=None, maxvalue=None):
        """
        Setup MappingEditor:
        return False if data is not supported, True otherwise
        """
        icon = ima.icon('larray')
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Session viewer") if readonly else _("Session editor")
        if readonly:
            title += ' (' + _('read only') + ')'
        self.title = title
        self.setWindowTitle(self.title)

        self.statusBar().showMessage("Welcome to the LArray Viewer", 4000)

        widget = QWidget()
        self.setCentralWidget(widget)

        layout = QVBoxLayout()
        widget.setLayout(layout)

        self._listwidget = QListWidget(self)
        self._listwidget.currentItemChanged.connect(self.on_item_changed)
        self._listwidget.setMinimumWidth(45)

        del_item_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self._listwidget)
        del_item_shortcut.activated.connect(self.delete_current_item)

        self.data = Session()
        self.arraywidget = ArrayEditorWidget(self, zeros(0), readonly)

        if qtconsole_available:
            # Create an in-process kernel
            kernel_manager = QtInProcessKernelManager()
            kernel_manager.start_kernel(show_banner=False)
            kernel = kernel_manager.kernel

            kernel.shell.run_cell('from larray import *')
            text_formatter = kernel.shell.display_formatter.formatters['text/plain']

            def void_formatter(array, *args, **kwargs):
                return ''

            for type_ in DISPLAY_IN_GRID:
                text_formatter.for_type(type_, void_formatter)

            self.kernel = kernel

            kernel_client = kernel_manager.client()
            kernel_client.start_channels()

            ipython_widget = RichJupyterWidget()
            ipython_widget.kernel_manager = kernel_manager
            ipython_widget.kernel_client = kernel_client
            ipython_widget.executed.connect(self.ipython_cell_executed)
            ipython_widget._display_banner = False

            self.eval_box = ipython_widget
            self.eval_box.setMinimumHeight(20)

            arraywidget = self.arraywidget
            if not readonly:
                # Buttons configuration
                btn_layout = QHBoxLayout()
                btn_layout.addStretch()

                bbox = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Discard)

                apply_btn = bbox.button(QDialogButtonBox.Apply)
                apply_btn.clicked.connect(self.apply_changes)

                discard_btn = bbox.button(QDialogButtonBox.Discard)
                discard_btn.clicked.connect(self.discard_changes)

                btn_layout.addWidget(bbox)

                arraywidget_layout = QVBoxLayout()
                arraywidget_layout.addWidget(self.arraywidget)
                arraywidget_layout.addLayout(btn_layout)

                # you cant add a layout directly in a splitter, so we have to wrap it in a widget
                arraywidget = QWidget()
                arraywidget.setLayout(arraywidget_layout)

            right_panel_widget = QSplitter(Qt.Vertical)
            right_panel_widget.addWidget(arraywidget)
            right_panel_widget.addWidget(self.eval_box)
            right_panel_widget.setSizes([90, 10])
        else:
            self.eval_box = QLineEdit()
            self.eval_box.returnPressed.connect(self.line_edit_update)

            right_panel_layout = QVBoxLayout()
            right_panel_layout.addWidget(self.arraywidget)
            right_panel_layout.addWidget(self.eval_box)

            # you cant add a layout directly in a splitter, so we have to wrap
            # it in a widget
            right_panel_widget = QWidget()
            right_panel_widget.setLayout(right_panel_layout)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self._listwidget)
        main_splitter.addWidget(right_panel_widget)
        main_splitter.setSizes([10, 90])
        main_splitter.setCollapsible(1, False)

        layout.addWidget(main_splitter)

        self._listwidget.setCurrentRow(0)

        self.resize(800, 600)
        self.setMinimumSize(400, 300)

        # Make the dialog act as a window
        self.setWindowFlags(Qt.Window)

        # check if reopen last opened file
        if data is REOPEN_LAST_FILE:
            if len(QSettings().value("recentFileList")) > 0:
                data = self.recent_file_actions[0].data()
            else:
                data = Session()

        # load file if any
        if isinstance(data, str):
            if os.path.isfile(data):
                self._open_file(data)
            else:
                QMessageBox.critical(self, "Error", "File {} could not be found".format(data))
                self.new()
        # convert input data to Session if not
        else:
            self.data = data if isinstance(data, Session) else Session(data)
            if qtconsole_available:
                self.kernel.shell.push(dict(self.data.items()))
            arrays = [k for k, v in self.data.items() if self._display_in_grid(k, v)]
            self.add_list_items(arrays)

        return True

    def _reset(self):
        self.data = Session()
        self._listwidget.clear()
        self.current_array = None
        self.current_array_name = None
        if qtconsole_available:
            self.kernel.shell.reset()
            self.kernel.shell.run_cell('from larray import *')
            self.ipython_cell_executed()
        else:
            self.eval_box.setText('None')
            self.line_edit_update()

    def setup_menu_bar(self):
        """Setup menu bar"""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')

        file_menu.addAction(create_action(self, _('&New'), shortcut="Ctrl+N", triggered=self.new))
        file_menu.addAction(create_action(self, _('&Open'), shortcut="Ctrl+O", triggered=self.open,
                                          statustip=_('Load session from file')))
        file_menu.addAction(create_action(self, _('&Save'), shortcut="Ctrl+S", triggered=self.save,
                                          statustip=_('Save all arrays as a session in a file')))
        file_menu.addAction(create_action(self, _('Save &As'), triggered=self.save_as,
                                          statustip=_('Save all arrays as a session in a file')))

        recent_files_menu = file_menu.addMenu("Open &Recent")
        for action in self.recent_file_actions:
            action.setVisible(False)
            action.triggered.connect(self.open_recent_file)
            recent_files_menu.addAction(action)
        self.update_recent_file_actions()
        recent_files_menu.addSeparator()
        recent_files_menu.addAction(create_action(self, _('&Clear List'), triggered=self._clear_recent_files))

        file_menu.addSeparator()
        file_menu.addAction(create_action(self, _('&Quit'), shortcut="Ctrl+Q", triggered=self.close))

        help_menu = menu_bar.addMenu('&Help')
        help_menu.addAction(create_action(self, _('Online &Documentation'), shortcut="Ctrl+H",
                                          triggered=self.open_documentation))
        help_menu.addAction(create_action(self, _('Online &Tutorial'), triggered=self.open_tutorial))
        help_menu.addAction(create_action(self, _('Online Objects and Functions (API) &Reference'),
                                          triggered=self.open_api_documentation))

    def add_list_item(self, name):
        listitem = QListWidgetItem(self._listwidget)
        listitem.setText(name)
        value = self.data[name]
        if isinstance(value, LArray):
            listitem.setToolTip(str(value.info))

    def add_list_items(self, names):
        for name in names:
            self.add_list_item(name)

    def delete_list_item(self, to_delete):
        deleted_items = self._listwidget.findItems(to_delete, Qt.MatchExactly)
        assert len(deleted_items) == 1
        deleted_item_idx = self._listwidget.row(deleted_items[0])
        self._listwidget.takeItem(deleted_item_idx)

    def select_list_item(self, to_display):
        changed_items = self._listwidget.findItems(to_display, Qt.MatchExactly)
        assert len(changed_items) == 1
        prev_selected = self._listwidget.selectedItems()
        assert len(prev_selected) <= 1
        # if the currently selected item (value) need to be refreshed (e.g it was modified)
        if prev_selected and prev_selected[0] == changed_items[0]:
            # we need to update the array widget explicitly
            self.set_current_array(self.data[to_display], to_display)
        else:
            # for some reason, on_item_changed is not triggered when no item was selected
            if not prev_selected:
                self.set_current_array(self.data[to_display], to_display)
            self._listwidget.setCurrentItem(changed_items[0])

    def update_mapping(self, value):
        # XXX: use ordered set so that the order is non-random if the underlying container is ordered?
        keys_before = set(self.data.keys())
        keys_after = set(value.keys())
        # contains both new and updated keys (but not deleted keys)
        changed_keys = [k for k in keys_after if value[k] is not self.data.get(k)]

        # when a key is re-assigned, it can switch from being displayable to non-displayable or vice versa
        displayable_keys_before = set(k for k in keys_before if self._display_in_grid(k, self.data[k]))
        displayable_keys_after = set(k for k in keys_after if self._display_in_grid(k, value[k]))
        deleted_displayable_keys = displayable_keys_before - displayable_keys_after
        new_displayable_keys = displayable_keys_after - displayable_keys_before
        # this can contain more keys than new_displayble_keys (because of existing keys which changed value)
        changed_displayable_keys = [k for k in changed_keys if self._display_in_grid(k, value[k])]

        # 1) update session/mapping
        # a) deleted old keys
        for k in keys_before - keys_after:
            del self.data[k]
        # b) add new/modify existing keys
        for k in changed_keys:
            self.data[k] = value[k]

        # 2) update list widget
        for k in deleted_displayable_keys:
            self.delete_list_item(k)
        self.add_list_items(new_displayable_keys)

        # 3) mark session as dirty if needed
        if len(changed_displayable_keys) > 0 or deleted_displayable_keys:
            self._unsaved_modifications = True

        # 4) change displayed array in the array widget
        # only display first result if there are more than one
        to_display = changed_displayable_keys[0] if changed_displayable_keys else None
        if to_display is not None:
            self.select_list_item(to_display)
        return to_display

    def delete_current_item(self):
        current_item = self._listwidget.currentItem()
        name = str(current_item.text())
        del self.data[name]
        if qtconsole_available:
            self.kernel.shell.del_var(name)
        self._listwidget.takeItem(self._listwidget.row(current_item))

    def line_edit_update(self):
        import larray as la
        s = self.eval_box.text()
        if assignment_pattern.match(s):
            context = self.data._objects.copy()
            exec(s, la.__dict__, context)
            varname = self.update_mapping(context)
            if varname is not None:
                self.expressions[varname] = s
        else:
            self.view_expr(eval(s, la.__dict__, self.data))

    def view_expr(self, array):
        self._listwidget.clearSelection()
        self.set_current_array(array, '<expr>')

    def _display_in_grid(self, k, v):
        return not k.startswith('__') and isinstance(v, DISPLAY_IN_GRID)

    def ipython_cell_executed(self):
        user_ns = self.kernel.shell.user_ns
        ip_keys = set(['In', 'Out', '_', '__', '___',
                       '__builtin__',
                       '_dh', '_ih', '_oh', '_sh', '_i', '_ii', '_iii',
                       'exit', 'get_ipython', 'quit'])
        # '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__',
        clean_ns_keys = set([k for k, v in user_ns.items() if not history_vars_pattern.match(k)]) - ip_keys
        clean_ns = {k: v for k, v in user_ns.items() if k in clean_ns_keys}

        # user_ns['_i'] is not updated yet (refers to the -2 item)
        # 'In' and '_ih' point to the same object (but '_ih' is supposed to be the non-overridden one)
        cur_input_num = len(user_ns['_ih']) - 1
        last_input = user_ns['_ih'][-1]
        if setitem_pattern.match(last_input):
            m = setitem_pattern.match(last_input)
            varname = m.group(1)
            # otherwise it should have failed at this point, but let us be sure
            if varname in clean_ns:
                if self._display_in_grid(varname, clean_ns[varname]):
                    self.select_list_item(varname)
        else:
            # not setitem => assume expr or normal assignment
            if last_input in clean_ns:
                # the name exists in the session (variable)
                if self._display_in_grid('', self.data[last_input]):
                    # select and display it
                    self.select_list_item(last_input)
            else:
                # any statement can contain a call to a function which updates globals
                self.update_mapping(clean_ns)

                # if the statement produced any output (probably because it is a simple expression), display it.

                # _oh and Out are supposed to be synonyms but "_ih" is supposed to be the non-overridden one.
                # It would be easier to use '_' instead but that refers to the last output, not the output of the
                # last command. Which means that if the last command did not produce any output, _ is not modified.
                cur_output = user_ns['_oh'].get(cur_input_num)
                if cur_output is not None:
                    if self._display_in_grid('_', cur_output):
                        self.view_expr(cur_output)

                    if isinstance(cur_output, matplotlib.axes.Subplot) and 'inline' not in matplotlib.get_backend():
                        show_figure(self, cur_output.figure)

    def on_item_changed(self, curr, prev):
        if curr is not None:
            name = str(curr.text())
            array = self.data[name]
            self.set_current_array(array, name)
            expr = self.expressions.get(name, name)
            if qtconsole_available:
                # this does not work because it updates the NEXT input, not the
                # current one (it is supposed to be called from within the console)
                # self.kernel.shell.set_next_input(expr, replace=True)
                # self.kernel_client.input(expr)
                pass
            else:
                self.eval_box.setText(expr)

    def update_title(self):
        array = self.current_array
        name = self.current_array_name
        title = []
        if isinstance(array, LArray):
            # current file (if not None)
            if self.current_file is not None:
                if os.path.isdir(self.current_file):
                    title = ['{}/{}.csv'.format(self.current_file, name)]
                else:
                    title = [self.current_file]
            # array info
            axes = array.axes
            axes_info = ' x '.join("%s (%d)" % (display_name, len(axis))
                                   for display_name, axis
                                   in zip(axes.display_names, axes))
            title += [(name + ': ' + axes_info) if name else axes_info]
        # name of non-LArray displayed item (if not None)
        elif name:
            title = [name]
        # extra info
        title += [self.title]
        self.setWindowTitle(' - '.join(title))

    def set_current_array(self, array, name):
        self.current_array = array
        self.current_array_name = name
        self.arraywidget.set_data(array)
        self.update_title()

    def _add_arrays(self, arrays):
        for k, v in arrays.items():
            self.data[k] = v
            self.add_list_item(k)
        if qtconsole_available:
            self.kernel.shell.push(dict(arrays))

    def _is_unsaved_modifications(self):
        if self.arraywidget.model.readonly:
            return False
        else:
            return self.arraywidget.dirty or self._unsaved_modifications

    def _ask_to_save_if_unsaved_modifications(self):
        """
        Returns
        -------
        bool
            whether or not the process should continue
        """
        if self._is_unsaved_modifications():
            ret = QMessageBox.warning(self, "Warning", "The data has been modified.\nDo you want to save your changes?",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if ret == QMessageBox.Save:
                self.apply_changes()
                return self.save()
            elif ret == QMessageBox.Cancel:
                return False
            else:
                return True
        else:
            return True

    def new(self):
        if self._ask_to_save_if_unsaved_modifications():
            self._reset()
            self.arraywidget.set_data(zeros(0))
            self.set_current_file(None)
            self._unsaved_modifications = False
            self.statusBar().showMessage("Viewer has been reset", 4000)

    def _open_file(self, filepath):
        self._reset()
        session = Session()
        if '.csv' in filepath:
            filepath = [filepath]
        if isinstance(filepath, (list, tuple)):
            session.load(None, filepath)
            dirname = os.path.dirname(filepath[0])
            basenames = [os.path.basename(fpath) for fpath in filepath]
            self.set_current_file(dirname)
            self.statusBar().showMessage("CSV files {} loaded".format(' ,'.join(basenames)), 4000)
        else:
            session.load(filepath)
            self.set_current_file(filepath)
            self.statusBar().showMessage("File {} loaded".format(os.path.basename(filepath)), 4000)
        self._add_arrays(session)
        self._listwidget.setCurrentRow(0)
        self._unsaved_modifications = False

    def open(self):
        if self._ask_to_save_if_unsaved_modifications():
            filter = "All (*.xls *xlsx *.h5 *.csv);;Excel Files (*.xls *xlsx);;HDF Files (*.h5);;CSV Files (*.csv)"
            res = QFileDialog.getOpenFileNames(self, filter=filter)
            # Qt5 returns a tuple (filepaths, '') instead of a string
            filepaths = res[0] if PYQT5 else res
            if len(filepaths) >= 1:
                if all(['.csv' in filepath for filepath in filepaths]):
                    self._open_file(filepaths)
                elif len(filepaths) == 1:
                    self._open_file(filepaths[0])
                else:
                    QMessageBox.critical(self, "Error",
                                         "Only several CSV files can be loaded in the same time")

    def open_recent_file(self):
        if self._ask_to_save_if_unsaved_modifications():
            action = self.sender()
            if action:
                filepath = action.data()
                if os.path.exists(filepath):
                    self._open_file(filepath)
                else:
                    QMessageBox.warning(self, "Warning", "File {} could not be found".format(filepath))

    def _save_data(self, filepath):
        session = Session({k: v for k, v in self.data.items() if self._display_in_grid(k, v)})
        session.save(filepath)
        self.set_current_file(filepath)
        self._unsaved_modifications = False
        self.statusBar().showMessage("Arrays saved in file {}".format(filepath), 4000)

    def save(self):
        """
        Returns
        -------
        bool
            whether or not the data was actually saved
        """
        if self.current_file is not None:
            self._save_data(self.current_file)
            return True
        else:
            return self.save_as()

    def save_as(self):
        # TODO: use filter
        dialog = QFileDialog(self)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        accepted = dialog.exec_() == QDialog.Accepted
        if accepted:
            self._save_data(dialog.selectedFiles()[0])
        return accepted

    def open_documentation(self):
        QDesktopServices.openUrl(QUrl("http://larray.readthedocs.io/en/stable/"))

    def open_tutorial(self):
        QDesktopServices.openUrl(QUrl("http://larray.readthedocs.io/en/stable/notebooks/LArray_intro.html"))

    def open_api_documentation(self):
        QDesktopServices.openUrl(QUrl("http://larray.readthedocs.io/en/stable/api.html"))

    def set_current_file(self, filepath):
        self.update_recent_files([filepath])
        self.current_file = filepath
        self.update_title()

    def update_recent_files(self, filepaths):
        settings = QSettings()
        files = settings.value("recentFileList")
        for filepath in filepaths:
            if filepath is not None:
                if filepath in files:
                    files.remove(filepath)
                files = [filepath] + files
        settings.setValue("recentFileList", files[:self.MAX_RECENT_FILES])
        self.update_recent_file_actions()

    def _clear_recent_files(self):
        settings = QSettings()
        settings.setValue("recentFileList", [])
        self.update_recent_file_actions()

    def update_recent_file_actions(self):
        settings = QSettings()
        recent_files = settings.value("recentFileList")
        if recent_files is None:
            recent_files = []

        # zip will iterate up to the shortest of the two
        for filepath, action in zip(recent_files, self.recent_file_actions):
            action.setText(os.path.basename(filepath))
            action.setStatusTip(filepath)
            action.setData(filepath)
            action.setVisible(True)
        # if we have less recent recent files than actions, hide the remaining actions
        for action in self.recent_file_actions[len(recent_files):]:
            action.setVisible(False)

    def closeEvent(self, event):
        if self._ask_to_save_if_unsaved_modifications():
            event.accept()
        else:
            event.ignore()

    def apply_changes(self):
        # update _unsaved_modifications only if 1 or more changes have been applied
        if len(self.arraywidget.model.changes) > 0:
            self._unsaved_modifications = True
        self.arraywidget.accept_changes()

    def discard_changes(self):
        self.arraywidget.reject_changes()

    def get_value(self):
        """Return modified array -- this is *not* a copy"""
        # It is import to avoid accessing Qt C++ object as it has probably
        # already been destroyed, due to the Qt.WA_DeleteOnClose attribute
        return self.data


class ArrayEditor(QDialog):
    """Array Editor Dialog"""
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.data = None
        self.arraywidget = None

    def setup_and_check(self, data, title='', readonly=False,
                        xlabels=None, ylabels=None,
                        minvalue=None, maxvalue=None):
        """
        Setup ArrayEditor:
        return False if data is not supported, True otherwise
        """
        if np.isscalar(data):
            readonly = True
        if isinstance(data, LArray):
            axes_info = ' x '.join("%s (%d)" % (display_name, len(axis))
                                   for display_name, axis
                                   in zip(data.axes.display_names, data.axes))
            title = (title + ': ' + axes_info) if title else axes_info

        self.data = data
        layout = QGridLayout()
        self.setLayout(layout)

        icon = ima.icon('larray')
        if icon is not None:
            self.setWindowIcon(icon)

        if not title:
            title = _("Array viewer") if readonly else _("Array editor")
        if readonly:
            title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)
        self.resize(800, 600)
        self.setMinimumSize(400, 300)

        self.arraywidget = ArrayEditorWidget(self, data, readonly, xlabels, ylabels,
                                             minvalue=minvalue, maxvalue=maxvalue)
        layout.addWidget(self.arraywidget, 1, 0)

        # Buttons configuration
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        # not using a QDialogButtonBox with standard Ok/Cancel buttons
        # because that makes it impossible to disable the AutoDefault on them
        # (Enter always "accepts"/close the dialog) which is annoying for edit()
        if readonly:
            close_button = QPushButton("Close")
            close_button.clicked.connect(self.reject)
            close_button.setAutoDefault(False)
            btn_layout.addWidget(close_button)
        else:
            ok_button = QPushButton("&OK")
            ok_button.clicked.connect(self.accept)
            ok_button.setAutoDefault(False)
            btn_layout.addWidget(ok_button)
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(self.reject)
            cancel_button.setAutoDefault(False)
            btn_layout.addWidget(cancel_button)
        # r_button = QPushButton("resize")
        # r_button.clicked.connect(self.resize_to_contents)
        # btn_layout.addWidget(r_button)
        layout.addLayout(btn_layout, 2, 0)

        # Make the dialog act as a window
        self.setWindowFlags(Qt.Window)
        return True

    def autofit_columns(self):
        self.arraywidget.view.autofit_columns()

    @Slot()
    def accept(self):
        """Reimplement Qt method"""
        self.arraywidget.accept_changes()
        QDialog.accept(self)

    @Slot()
    def reject(self):
        """Reimplement Qt method"""
        self.arraywidget.reject_changes()
        QDialog.reject(self)

    def get_value(self):
        """Return modified array -- this is *not* a copy"""
        # It is import to avoid accessing Qt C++ object as it has probably
        # already been destroyed, due to the Qt.WA_DeleteOnClose attribute
        return self.data
