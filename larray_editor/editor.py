import os
import re
import matplotlib
import numpy as np

from larray import LArray, Session, zeros, empty
from larray_editor.utils import (PY2, PYQT5, _, create_action, show_figure, ima, commonpath, dependencies,
                                 get_versions, get_documentation_url, urls)
from larray_editor.arraywidget import ArrayEditorWidget
from qtpy.QtCore import Qt, QSettings, QUrl, Slot
from qtpy.QtGui import QDesktopServices, QKeySequence
from qtpy.QtWidgets import (QMainWindow, QWidget, QListWidget, QListWidgetItem, QSplitter, QFileDialog, QPushButton,
                            QDialogButtonBox, QAction, QShortcut, QHBoxLayout, QVBoxLayout, QGridLayout, QLineEdit,
                            QCheckBox, QMessageBox, QDialog, QInputDialog, QLabel, QGroupBox, QRadioButton)

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

        # to handle recently opened data/script files
        settings = QSettings()
        # data files
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
        self._title = title
        self.setWindowTitle(title)

        self.statusBar().showMessage("Welcome to the LArray Viewer", 4000)

        widget = QWidget()
        self.setCentralWidget(widget)

        layout = QVBoxLayout()
        widget.setLayout(layout)

        self._listwidget = QListWidget(self)
        # this is a bit more reliable than currentItemChanged which is not emitted when no item was selected before
        self._listwidget.itemSelectionChanged.connect(self.on_selection_changed)
        self._listwidget.setMinimumWidth(45)

        del_item_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self._listwidget)
        del_item_shortcut.activated.connect(self.delete_current_item)

        self.data = Session()
        self.arraywidget = ArrayEditorWidget(self, readonly=readonly)
        self.arraywidget.model_data.dataChanged.connect(self.data_changed)

        if qtconsole_available:
            # Create an in-process kernel
            kernel_manager = QtInProcessKernelManager()
            kernel_manager.start_kernel(show_banner=False)
            kernel = kernel_manager.kernel

            # TODO: use self._reset() instead
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
                arraywidget_layout.setContentsMargins(0, 0, 0, 0)

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
        self._listwidget.setCurrentRow(0)
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

        #################
        #   FILE MENU   #
        #################
        file_menu = menu_bar.addMenu('&File')

        #===============#
        #      NEW      #
        #===============#
        file_menu.addAction(create_action(self, _('&New'), shortcut="Ctrl+N", triggered=self.new))
        file_menu.addSeparator()
        #===============#
        #     DATA      #
        #===============#
        file_menu.addSeparator()
        file_menu.addAction(create_action(self, _('&Open Data'), shortcut="Ctrl+O", triggered=self.open_data,
                                          statustip=_('Load session from file')))
        file_menu.addAction(create_action(self, _('&Save Data'), shortcut="Ctrl+S", triggered=self.save_data,
                                          statustip=_('Save all arrays as a session in a file')))
        file_menu.addAction(create_action(self, _('Save Data &As'), triggered=self.save_data_as,
                                          statustip=_('Save all arrays as a session in a file')))
        recent_files_menu = file_menu.addMenu("Open &Recent Data")
        for action in self.recent_file_actions:
            action.setVisible(False)
            action.triggered.connect(self.open_recent_file)
            recent_files_menu.addAction(action)
        self.update_recent_file_actions()
        recent_files_menu.addSeparator()
        recent_files_menu.addAction(create_action(self, _('&Clear List'), triggered=self._clear_recent_files))
        #===============#
        #    EXAMPLES   #
        #===============#
        file_menu.addSeparator()
        file_menu.addAction(create_action(self, _('&Load Example Dataset'), triggered=self.load_example))
        #===============#
        #    SCRIPTS    #
        #===============#
        file_menu.addSeparator()
        if qtconsole_available:
            file_menu.addAction(create_action(self, _('&Load from Script'), shortcut="Ctrl+Shift+O",
                                              triggered=self.load_script, statustip=_('Load script from file')))
            file_menu.addAction(create_action(self, _('&Save Command History To Script'), shortcut="Ctrl+Shift+S",
                                              triggered=self.save_script, statustip=_('Save command history in a file')))
        #===============#
        #     QUIT      #
        #===============#
        file_menu.addSeparator()
        file_menu.addAction(create_action(self, _('&Quit'), shortcut="Ctrl+Q", triggered=self.close))

        #################
        #   HELP MENU   #
        #################
        help_menu = menu_bar.addMenu('&Help')

        #===============#
        # DOCUMENTATION #
        #===============#
        help_menu.addAction(create_action(self, _('Online &Documentation'), shortcut="Ctrl+H",
                                          triggered=self.open_documentation))
        help_menu.addAction(create_action(self, _('Online &Tutorial'), triggered=self.open_tutorial))
        help_menu.addAction(create_action(self, _('Online Objects and Functions (API) &Reference'),
                                          triggered=self.open_api_documentation))
        #======================#
        # ISSUES/GOOGLE GROUPS #
        #======================#
        help_menu.addSeparator()
        report_issue_menu = help_menu.addMenu("Report &Issue...")
        report_issue_menu.addAction(create_action(self, _('Report &Editor Issue...'),
                                                  triggered=self.report_issue('editor')))
        report_issue_menu.addAction(create_action(self, _('Report &LArray Issue...'),
                                                  triggered=self.report_issue('larray')))
        report_issue_menu.addAction(create_action(self, _('Report &LArray Eurostat Issue...'),
                                                  triggered=self.report_issue('larray_eurostat')))
        help_menu.addAction(create_action(self, _('&Users Discussion...'), triggered=self.open_users_group))
        help_menu.addAction(create_action(self, _('New Releases And &Announces Mailing List...'),
                                          triggered=self.open_announce_group))
        #=================#
        #       ABOUT     #
        #=================#
        help_menu.addSeparator()
        help_menu.addAction(create_action(self, _('&About'), triggered=self.about))

    def data_changed(self):
        # We do not set self._unsaved_modifications to True because if users click on `Discard` button
        # (which calls reject_changes) or choose to display another array, all temporary changes are lost.
        # `update_title` relies on _is_unsaved_modifications() which checks both self._unsaved_modifications
        # and self.arraywidget.dirty
        self.update_title()

    @property
    def unsaved_modifications(self):
        return self._unsaved_modifications

    @unsaved_modifications.setter
    def unsaved_modifications(self, unsaved_modifications):
        self._unsaved_modifications = unsaved_modifications
        self.update_title()

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
            self._listwidget.setCurrentItem(changed_items[0])

    def update_mapping(self, value):
        # XXX: use ordered set so that the order is non-random if the underlying container is ordered?
        keys_before = set(self.data.keys())
        keys_after = set(value.keys())
        # Contains both new and keys for which the object id changed (but not deleted keys nor inplace modified keys).
        # Inplace modified arrays should be already handled in ipython_cell_executed by the setitem_pattern.
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
            self.unsaved_modifications = True

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
        self.unsaved_modifications = True
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
                    # XXX: this completely refreshes the array, including detecting scientific & ndigits, which might
                    # not be what we want in this case
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
                # this will select (or refresh) the "first" changed array
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

    def on_selection_changed(self):
        selected = self._listwidget.selectedItems()
        if selected:
            assert len(selected) == 1
            selected_item = selected[0]
            assert isinstance(selected_item, QListWidgetItem)
            name = str(selected_item.text())
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
        name = self.current_array_name if self.current_array_name is not None else ''

        unsaved_marker = '*' if self._is_unsaved_modifications() else ''
        if self.current_file is not None:
            basename = os.path.basename(self.current_file)
            if os.path.isdir(self.current_file):
                assert not name.endswith('.csv')
                fname = os.path.join(basename, '{}.csv'.format(name))
                name = ''
            else:
                fname = basename
        else:
            fname = '<new>'
        title = ['{}{}'.format(unsaved_marker, fname)]

        if array is not None:
            dtype = array.dtype.name
            # current file (if not None)
            if isinstance(array, LArray):
                # array info
                shape = ['{} ({})'.format(display_name, len(axis))
                         for display_name, axis in zip(array.axes.display_names, array.axes)]
            else:
                # if it's not an LArray, it must be a Numpy ndarray
                assert isinstance(array, np.ndarray)
                shape = [str(length) for length in array.shape]
            # name + shape + dtype
            array_info = ' x '.join(shape) + ' [{}]'.format(dtype)
            if name:
                title += [name + ': ' + array_info]
            else:
                title += [array_info]

        # extra info
        title += [self._title]
        # set title
        self.setWindowTitle(' - '.join(title))

    def set_current_array(self, array, name):
        # we should NOT check that "array is not self.current_array" because this method is also called to
        # refresh the widget value because of an inplace setitem
        self.current_array = array
        self.arraywidget.set_data(array)
        self.current_array_name = name
        self.update_title()

    def set_current_file(self, filepath):
        self.update_recent_files([filepath])
        self.current_file = filepath
        self.update_title()

    def _add_arrays(self, arrays):
        for k, v in arrays.items():
            self.data[k] = v
            self.add_list_item(k)
        if qtconsole_available:
            self.kernel.shell.push(dict(arrays))

    def _is_unsaved_modifications(self):
        if self.arraywidget.readonly:
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
                return self.save_data()
            elif ret == QMessageBox.Cancel:
                return False
            else:
                return True
        else:
            return True

    def closeEvent(self, event):
        if self._ask_to_save_if_unsaved_modifications():
            event.accept()
        else:
            event.ignore()

    def apply_changes(self):
        # update unsaved_modifications (and thus title) only if at least 1 change has been applied
        if self.arraywidget.dirty:
            self.unsaved_modifications = True
        self.arraywidget.accept_changes()

    def discard_changes(self):
        self.arraywidget.reject_changes()
        self.update_title()

    def get_value(self):
        """Return modified array -- this is *not* a copy"""
        # It is import to avoid accessing Qt C++ object as it has probably
        # already been destroyed, due to the Qt.WA_DeleteOnClose attribute
        return self.data

    #########################################
    #               FILE MENU               #
    #########################################

    def new(self):
        if self._ask_to_save_if_unsaved_modifications():
            self._reset()
            self.arraywidget.set_data(empty(0))
            self.set_current_file(None)
            self.unsaved_modifications = False
            self.statusBar().showMessage("Viewer has been reset", 4000)

    #================================#
    #  METHODS TO SAVE/LOAD SCRIPTS  #
    #================================#

    # See http://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-load
    # for more details
    def _load_script(self, filepath, lines, symbols):
        assert qtconsole_available
        try:
            cmd = []
            if lines:
                # -r <lines>: Specify lines or ranges of lines to load from the source.
                # Ranges could be specified as x..y (x-y) or in python-style x:y (x..(y-1)).
                # Both limits x and y can be left blank (meaning the beginning and end of the file, respectively).
                lines = lines.replace('..', '-')
                cmd += ['-r {}'.format(lines)]
            if symbols:
                # -s <symbols>: Specify function or classes to load from python source.
                cmd += ['-s {}'.format(symbols)]
            cmd += [filepath]
            self.kernel.shell.run_line_magic('load', ' '.join(cmd))
            self.ipython_cell_executed()
            self.update_recent_script_list(filepath)
        except Exception as e:
            QMessageBox.critical(self, "Error", "Cannot load script file {}:\n{}"
                                 .format(os.path.basename(filepath), e))

    def load_script(self, filepath=None):
        # %save add automatically the extension .py if not present in passed filename
        dialog = QDialog(self)
        layout = QGridLayout()
        dialog.setLayout(layout)

        # filepath
        browse_label = QLabel("Source")
        browse_edit = QLineEdit()
        browse_edit.setPlaceholderText("filepath to or URL containing the python source")
        browse_button = QPushButton("Browse")
        if isinstance(filepath, str):
            browse_edit.setText(filepath)
        browse_filedialog = QFileDialog(self, filter="Python Script (*.py)")
        browse_filedialog.setFileMode(QFileDialog.ExistingFile)
        browse_button.clicked.connect(browse_filedialog.open)
        browse_filedialog.fileSelected.connect(browse_edit.setText)
        layout.addWidget(browse_label, 0, 0)
        layout.addWidget(browse_edit, 0, 1)
        layout.addWidget(browse_button, 0, 2)

        # lines / symbols
        group_box = QGroupBox()
        group_box_layout = QGridLayout()
        # all lines
        radio_button_all_lines = QRadioButton("Load all file")
        radio_button_all_lines.setChecked(True)
        group_box_layout.addWidget(radio_button_all_lines, 0, 0)
        # specific lines
        radio_button_specific_lines = QRadioButton("Load specific lines")
        radio_button_specific_lines.setToolTip("Selected (ranges of) lines to load must be separated with "
                                               "whitespaces.\nRanges could be specified as x..y (x-y) or in "
                                               "python-style x:y (x..(y-1)).")
        lines_edit = QLineEdit()
        lines_edit.setPlaceholderText("1 4..6 8")
        lines_edit.setEnabled(False)
        radio_button_specific_lines.toggled.connect(lines_edit.setEnabled)
        group_box_layout.addWidget(radio_button_specific_lines, 1, 0)
        group_box_layout.addWidget(lines_edit, 1, 1)
        # specific symbols (variables, functions and classes)
        radio_button_symbols = QRadioButton("Load symbols")
        symbols_edit = QLineEdit()
        symbols_edit.setPlaceholderText("variables or functions separated by commas")
        symbols_edit.setEnabled(False)
        radio_button_symbols.toggled.connect(symbols_edit.setEnabled)
        group_box_layout.addWidget(radio_button_symbols, 2, 0)
        group_box_layout.addWidget(symbols_edit, 2, 1)
        # set layout
        group_box.setLayout(group_box_layout)
        layout.addWidget(group_box, 1, 0, 1, 3)

        clear_session_checkbox = QCheckBox("Clear session before to load")
        clear_session_checkbox.setChecked(False)
        layout.addWidget(clear_session_checkbox, 2, 0, 1, 3)

        # accept/reject
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(dialog.accept)
        bbox.rejected.connect(dialog.reject)
        layout.addWidget(bbox, 3, 0, 1, 3)

        # open dialog
        ret = dialog.exec_()
        if ret == QDialog.Accepted:
            filepath = browse_edit.text()
            if radio_button_specific_lines.isChecked():
                lines, symbols = lines_edit.text(), ''
            elif radio_button_symbols.isChecked():
                lines, symbols = '', symbols_edit.text()
            else:
                lines, symbols = '', ''
            if clear_session_checkbox.isChecked():
                self._reset()
            self._load_script(filepath, lines, symbols)

    def open_recent_script(self):
        if self._ask_to_save_if_unsaved_modifications():
            action = self.sender()
            if action:
                filepath = action.data()
                if os.path.exists(filepath):
                    self.load_script(filepath)
                else:
                    QMessageBox.warning(self, "Warning", "File {} could not be found".format(filepath))

    def update_recent_script_list(self, filepath):
        settings = QSettings()
        scripts = settings.value("recentScriptList")
        if filepath is not None and filepath in scripts:
            scripts.remove(filepath)
        scripts = [filepath] + scripts
        settings.setValue("recentScriptList", scripts[:self.MAX_RECENT_FILES])
        self.update_recent_script_actions()

    def _clear_recent_scripts(self):
        settings = QSettings()
        settings.setValue("recentScriptList", [])
        self.update_recent_script_actions()

    def update_recent_script_actions(self):
        settings = QSettings()
        recent_scripts = settings.value("recentScriptList")
        if recent_scripts is None:
            recent_scripts = []

        # zip will iterate up to the shortest of the two
        for filepath, action in zip(recent_scripts, self.recent_script_actions):
            action.setText(os.path.basename(filepath))
            action.setStatusTip(filepath)
            action.setData(filepath)
            action.setVisible(True)
        # if we have less recent recent files than actions, hide the remaining actions
        for action in self.recent_script_actions[len(recent_scripts):]:
            action.setVisible(False)

    def _save_script(self, filepath, lines, overwrite):
        assert qtconsole_available
        try:
            # -f: force overwrite. If file exists, %save will prompt for overwrite unless -f is given.
            # -a: append to the file instead of overwriting it.
            overwrite = '-f' if overwrite else '-a'
            if lines:
                lines = lines.replace('..', '-')
            else:
                lines = '1-{}'.format(self.kernel.shell.execution_count)
            self.kernel.shell.run_line_magic('save', '{} {} {}'.format(overwrite, filepath, lines))
        except Exception as e:
            QMessageBox.critical(self, "Error", "Cannot save history as {}:\n{}"
                                 .format(os.path.basename(filepath), e))

    # See http://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-save
    # for more details
    def save_script(self):
        # %save add automatically the extension .py if not present in passed filename
        dialog = QDialog(self)
        layout = QGridLayout()
        dialog.setLayout(layout)

        # filepath
        browse_label = QLabel("Filepath")
        browse_edit = QLineEdit()
        browse_button = QPushButton("Browse")
        browse_filedialog = QFileDialog(self, filter="Python Script (*.py)")
        browse_button.clicked.connect(browse_filedialog.open)
        browse_filedialog.fileSelected.connect(browse_edit.setText)
        layout.addWidget(browse_label, 0, 0)
        layout.addWidget(browse_edit, 0, 1)
        layout.addWidget(browse_button, 0, 2)

        # lines
        group_box = QGroupBox()
        group_box_layout = QGridLayout()
        # all lines
        radio_button_all_lines = QRadioButton("Save all history")
        radio_button_all_lines.setChecked(True)
        group_box_layout.addWidget(radio_button_all_lines, 0, 0)
        # specific lines
        radio_button_specific_lines = QRadioButton("Save input lines")
        radio_button_specific_lines.setToolTip("Selected (ranges of) input lines must be separated with whitespaces.\n"
                                               "Ranges could be specified as x..y (x-y) or in python-style "
                                               "x:y (x..(y-1)).")
        lines_edit = QLineEdit()
        lines_edit.setPlaceholderText("1 4..6 8")
        lines_edit.setEnabled(False)
        radio_button_specific_lines.toggled.connect(lines_edit.setEnabled)
        group_box_layout.addWidget(radio_button_specific_lines, 1, 0)
        group_box_layout.addWidget(lines_edit, 1, 1)
        # set layout
        group_box.setLayout(group_box_layout)
        layout.addWidget(group_box, 1, 0, 1, 3)

        # overwrite/append to script
        group_box = QGroupBox()
        group_box_layout = QGridLayout()
        # overwrite
        radio_button_overwrite = QRadioButton("Overwrite file")
        radio_button_overwrite.setChecked(True)
        group_box_layout.addWidget(radio_button_overwrite, 0, 0)
        # append to
        radio_button_append = QRadioButton("Append to file")
        group_box_layout.addWidget(radio_button_append, 0, 1)
        # set layout
        group_box.setLayout(group_box_layout)
        layout.addWidget(group_box, 2, 0, 1, 3)

        # accept/reject
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(dialog.accept)
        bbox.rejected.connect(dialog.reject)
        layout.addWidget(bbox, 3, 0, 1, 3)

        # open dialog
        ret = dialog.exec_()
        if ret == QDialog.Accepted:
            filepath = browse_edit.text()
            if filepath == '':
                QMessageBox.warning(self, "Warning", "No file provided")
            else:
                if radio_button_specific_lines.isChecked():
                    lines = lines_edit.text()
                else:
                    lines = ''
                overwrite = radio_button_overwrite.isChecked()
                self._save_script(filepath, lines, overwrite)

    #=============================#
    #  METHODS TO SAVE/LOAD DATA  #
    #=============================#

    def _open_file(self, filepath):
        session = Session()
        # a list => .csv files. Possibly a single .csv file.
        if isinstance(filepath, (list, tuple)):
            fpaths = filepath
            if len(fpaths) == 1:
                common_fpath = os.path.dirname(fpaths[0])
            else:
                common_fpath = commonpath(fpaths)
            basenames = [os.path.basename(fpath) for fpath in fpaths]
            fnames = [os.path.relpath(fpath, common_fpath) for fpath in fpaths]

            names = [os.path.splitext(fname)[0] for fname in fnames]
            current_file_name = common_fpath
            display_name = ','.join(basenames)
            filepath = common_fpath
        else:
            names = None
            current_file_name = filepath
            display_name = os.path.basename(filepath)
        try:
            session.load(filepath, names)
            self._reset()
            self._add_arrays(session)
            self._listwidget.setCurrentRow(0)
            self.set_current_file(current_file_name)
            self.unsaved_modifications = False
            self.statusBar().showMessage("Loaded: {}".format(display_name), 4000)
        except Exception as e:
            QMessageBox.critical(self, "Error", "Something went wrong during load of file(s) {}:\n{}"
                                 .format(display_name, e))

    def open_data(self):
        if self._ask_to_save_if_unsaved_modifications():
            filter = "All (*.xls *xlsx *.h5 *.csv);;Excel Files (*.xls *xlsx);;HDF Files (*.h5);;CSV Files (*.csv)"
            res = QFileDialog.getOpenFileNames(self, filter=filter)
            # Qt5 returns a tuple (filepaths, '') instead of a string
            filepaths = res[0] if PYQT5 else res
            if len(filepaths) >= 1:
                if all(['.csv' in filepath for filepath in filepaths]):
                    # this means that even a single .csv file will be passed as a list (so that we can add arrays
                    # and save them as a directory).
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

    def _save_data(self, filepath):
        try:
            session = Session({k: v for k, v in self.data.items() if self._display_in_grid(k, v)})
            session.save(filepath)
            self.set_current_file(filepath)
            self.unsaved_modifications = False
            self.statusBar().showMessage("Arrays saved in file {}".format(filepath), 4000)
        except Exception as e:
            QMessageBox.critical(self, "Error", "Something went wrong during save in file {}:\n{}".format(filepath, e))

    def save_data(self):
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
            return self.save_data_as()

    def save_data_as(self):
        # TODO: use filter
        dialog = QFileDialog(self)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        accepted = dialog.exec_() == QDialog.Accepted
        if accepted:
            self._save_data(dialog.selectedFiles()[0])
        return accepted

    def load_example(self):
        if self._ask_to_save_if_unsaved_modifications():
            from larray.example import AVAILABLE_EXAMPLE_DATA
            dataset_names = sorted(AVAILABLE_EXAMPLE_DATA.keys())
            dataset_name, ok = QInputDialog.getItem(self, "load dataset example", "list of datasets examples",
                                                    dataset_names, 0, False)
            if ok and dataset_name:
                filepath = AVAILABLE_EXAMPLE_DATA[dataset_name]
                self._open_file(filepath)

    #########################################
    #               HELP MENU               #
    #########################################

    def open_documentation(self):
        QDesktopServices.openUrl(QUrl(get_documentation_url('doc_index')))

    def open_tutorial(self):
        QDesktopServices.openUrl(QUrl(get_documentation_url('doc_tutorial')))

    def open_api_documentation(self):
        QDesktopServices.openUrl(QUrl(get_documentation_url('doc_api')))

    def report_issue(self, package):
        def _report_issue(*args, **kwargs):
            if PY2:
                from urllib import quote
            else:
                from urllib.parse import quote

            versions = get_versions(package)
            issue_template = """\
## Description
**What steps will reproduce the problem?**
1. 
2. 
3.
 
**What is the expected output? What do you see instead?**


**Please provide any additional information below**


## Version and main components
* Python {python} on {system} {bitness:d}bits
"""
            issue_template += "* {package} {{{package}}}\n".format(package=package)
            for dep in dependencies[package]:
                issue_template += "* {dep} {{{dep}}}\n".format(dep=dep)
            issue_template = issue_template.format(**versions)

            url = QUrl(urls['new_issue_{}'.format(package)])
            if PYQT5:
                from qtpy.QtCore import QUrlQuery
                query = QUrlQuery()
                query.addQueryItem("body", quote(issue_template))
                url.setQuery(query)
            else:
                url.addEncodedQueryItem("body", quote(issue_template))
            QDesktopServices.openUrl(url)

        return _report_issue

    def open_users_group(self):
        QDesktopServices.openUrl(QUrl(urls['users_group']))

    def open_announce_group(self):
        QDesktopServices.openUrl(QUrl(urls['announce_group']))

    def about(self):
        """About Editor"""
        kwargs = get_versions('editor')
        kwargs.update(urls)
        message = """\
<p><b>LArray Editor</b> {editor}
<br>The Graphical User Interface for LArray
<p>Licensed under the terms of the <a href="{GPL3}">GNU General Public License Version 3</a>.
<p>Developed and maintained by the <a href="{fpb}">Federal Planning Bureau</a> (Belgium).
<p>&nbsp;
<p><b>Versions of underlying libraries</b>
<ul>
<li>Python {python} on {system} {bitness:d}bits</li>
"""
        for dep in dependencies['editor']:
            if kwargs[dep] != 'N/A':
                message += "<li>{dep} {{{dep}}}</li>\n".format(dep=dep)
        message += "</ul>"
        QMessageBox.about(self, _("About LArray Editor"), message.format(**kwargs))


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

    def setup_and_check(self, data, title='', readonly=False, minvalue=None, maxvalue=None):
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

        self.arraywidget = ArrayEditorWidget(self, data, readonly, minvalue=minvalue, maxvalue=maxvalue)
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
        self.arraywidget.autofit_columns()

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
