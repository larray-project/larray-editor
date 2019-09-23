import os
import re
import matplotlib
import numpy as np
import collections

from larray import LArray, Session, empty
from larray_editor.utils import (PY2, PYQT5, _, create_action, show_figure, ima, commonpath, dependencies,
                                 get_versions, get_documentation_url, urls, RecentlyUsedList)
from larray_editor.arraywidget import ArrayEditorWidget
from larray_editor.commands import EditSessionArrayCommand, EditCurrentArrayCommand

from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QDesktopServices, QKeySequence
from qtpy.QtWidgets import (QMainWindow, QWidget, QListWidget, QListWidgetItem, QSplitter, QFileDialog, QPushButton,
                            QDialogButtonBox, QShortcut, QHBoxLayout, QVBoxLayout, QGridLayout, QLineEdit, QUndoStack,
                            QCheckBox, QComboBox, QMessageBox, QDialog, QInputDialog, QLabel, QGroupBox, QRadioButton)

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

assignment_pattern = re.compile(r'[^\[\]]+[^=]=[^=].+')
setitem_pattern = re.compile(r'(.+)\[.+\][^=]=[^=].+')
history_vars_pattern = re.compile(r'_i?\d+')
# XXX: add all scalars except strings (from numpy or plain Python)?
# (long) strings are not handled correctly so should NOT be in this list
# tuple, list
DISPLAY_IN_GRID = (LArray, np.ndarray)


class AbstractEditor(QMainWindow):
    """Abstract Editor Window"""

    name = "Editor"

    def __init__(self, parent=None, editable=False, file_menu=False, help_menu=False):
        QMainWindow.__init__(self, parent)
        self._file_menu = file_menu
        self._edit_menu = editable
        self._help_menu = help_menu

        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.data = None
        self.arraywidget = None
        if editable:
            self.edit_undo_stack = QUndoStack(self)

    def setup_and_check(self, data, title='', readonly=False, caller_info=None, **kwargs):
        """Return False if data is not supported, True otherwise"""
        # set icon
        icon = ima.icon('larray')
        if icon is not None:
            self.setWindowIcon(icon)

        # set title
        if not title:
            title = _(self.name)
        if readonly:
            title += ' (' + _('read only') + ')'
        self._title = title
        self.setWindowTitle(title)

        # permanently display caller info in the status bar
        if caller_info is not None:
            caller_info = 'launched from file {} at line {}'.format(caller_info.filename, caller_info.lineno)
            self.statusBar().addPermanentWidget(QLabel(caller_info))
        # display welcome message
        self.statusBar().showMessage("Welcome to the {}".format(self.name), 4000)

        # set central widget
        widget = QWidget()
        self.setCentralWidget(widget)

        # setup central widget
        self._setup_and_check(widget, data, title, readonly, **kwargs)

        # resize
        self.resize(1000, 600)
        # This is more or less the minimum space required to display a 1D array
        self.setMinimumSize(300, 180)
        return True

    def setup_menu_bar(self):
        """Setup menu bar"""
        menu_bar = self.menuBar()
        if self._file_menu:
            self._setup_file_menu(menu_bar)
        if self._edit_menu:
            self._setup_edit_menu(menu_bar)
        if self._help_menu:
            self._setup_help_menu(menu_bar)

    def _setup_file_menu(self, menu_bar):
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(create_action(self, _('&Quit'), shortcut="Ctrl+Q", triggered=self.close))

    def _setup_edit_menu(self, menu_bar):
        if qtconsole_available:
            edit_menu = menu_bar.addMenu('&Edit')
            # UNDO
            undo_action = self.edit_undo_stack.createUndoAction(self, "&Undo")
            undo_action.setShortcuts(QKeySequence.Undo)
            undo_action.triggered.connect(self.update_title)
            edit_menu.addAction(undo_action)
            # REDO
            redo_action = self.edit_undo_stack.createRedoAction(self, "&Redo")
            redo_action.setShortcuts(QKeySequence.Redo)
            redo_action.triggered.connect(self.update_title)
            edit_menu.addAction(redo_action)

    def _setup_help_menu(self, menu_bar):
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

    def _update_title(self, title, array, name):
        if title is None:
            title = []

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

    def get_value(self):
        """Return modified array -- this is *not* a copy"""
        # It is import to avoid accessing Qt C++ object as it has probably
        # already been destroyed, due to the Qt.WA_DeleteOnClose attribute
        return self.data

    def _setup_and_check(self, widget, data, title, readonly, **kwargs):
        raise NotImplementedError()

    def update_title(self):
        raise NotImplementedError()


class MappingEditor(AbstractEditor):
    """Session Editor Dialog"""

    name = "Session Editor"

    def __init__(self, parent=None):
        AbstractEditor.__init__(self, parent, editable=True, file_menu=True, help_menu=True)

        # to handle recently opened data/script files
        self.recent_data_files = RecentlyUsedList("recentFileList", self, self.open_recent_file)
        self.recent_saved_scripts = RecentlyUsedList("recentSavedScriptList")
        self.recent_loaded_scripts = RecentlyUsedList("recentLoadedScriptList")

        self.current_file = None
        self.current_array = None
        self.current_array_name = None

        self._listwidget = None
        self.eval_box = None
        self.expressions = {}
        self.kernel = None
        self._unsaved_modifications = False

        self.setup_menu_bar()

    def _setup_and_check(self, widget, data, title, readonly, **kwargs):
        """Setup MappingEditor"""
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
        self.arraywidget.dataChanged.connect(self.push_changes)
        self.arraywidget.model_data.dataChanged.connect(self.update_title)

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

            right_panel_widget = QSplitter(Qt.Vertical)
            right_panel_widget.addWidget(self.arraywidget)
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

        # check if reopen last opened file
        if data is REOPEN_LAST_FILE:
            if len(self.recent_data_files.files) > 0:
                data = self.recent_data_file.files[0]
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
            self._push_data(data)

    def _push_data(self, data):
        self.data = data if isinstance(data, Session) else Session(data)
        if qtconsole_available:
            self.kernel.shell.push(dict(self.data.items()))
        arrays = [k for k, v in self.data.items() if self._display_in_grid(k, v)]
        self.add_list_items(arrays)
        self._listwidget.setCurrentRow(0)

    def _reset(self):
        self.data = Session()
        self._listwidget.clear()
        self.current_array = None
        self.current_array_name = None
        self.edit_undo_stack.clear()
        if qtconsole_available:
            self.kernel.shell.reset()
            self.kernel.shell.run_cell('from larray import *')
            self.ipython_cell_executed()
        else:
            self.eval_box.setText('None')
            self.line_edit_update()

    def _setup_file_menu(self, menu_bar):
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
        for action in self.recent_data_files.actions:
            recent_files_menu.addAction(action)
        recent_files_menu.addSeparator()
        recent_files_menu.addAction(create_action(self, _('&Clear List'), triggered=self.recent_data_files.clear))
        #===============#
        #    EXAMPLES   #
        #===============#
        file_menu.addSeparator()
        file_menu.addAction(create_action(self, _('&Load Example Dataset'), triggered=self.load_example))
        #===============#
        #    SCRIPTS    #
        #===============#
        if qtconsole_available:
            file_menu.addSeparator()
            file_menu.addAction(create_action(self, _('&Load from Script'), shortcut="Ctrl+Shift+O",
                                              triggered=self.load_script, statustip=_('Load script from file')))
            file_menu.addAction(create_action(self, _('&Save Command History To Script'), shortcut="Ctrl+Shift+S",
                                              triggered=self.save_script, statustip=_('Save command history in a file')))

        #===============#
        #     QUIT      #
        #===============#
        file_menu.addSeparator()
        file_menu.addAction(create_action(self, _('&Quit'), shortcut="Ctrl+Q", triggered=self.close))

    def push_changes(self, changes):
        self.edit_undo_stack.push(EditSessionArrayCommand(self, self.current_array_name, changes))

    @property
    def unsaved_modifications(self):
        return self.edit_undo_stack.canUndo() or self._unsaved_modifications

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
        ip_keys = {'In', 'Out', '_', '__', '___', '__builtin__', '_dh', '_ih', '_oh', '_sh', '_i', '_ii', '_iii',
                   'exit', 'get_ipython', 'quit'}
        # '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__',
        clean_ns = {k: v for k, v in user_ns.items() if k not in ip_keys and not history_vars_pattern.match(k)}

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

                    if isinstance(cur_output, collections.Iterable):
                        cur_output = np.ravel(cur_output)[0]

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
        name = self.current_array_name if self.current_array_name is not None else ''
        unsaved_marker = '*' if self.unsaved_modifications else ''
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

        array = self.current_array
        title = ['{}{}'.format(unsaved_marker, fname)]
        self._update_title(title, array, name)

    def set_current_array(self, array, name):
        # we should NOT check that "array is not self.current_array" because this method is also called to
        # refresh the widget value because of an inplace setitem
        self.current_array = array
        self.arraywidget.set_data(array)
        self.current_array_name = name
        self.update_title()

    def set_current_file(self, filepath):
        self.recent_data_files.add(filepath)
        self.current_file = filepath
        self.update_title()

    def _ask_to_save_if_unsaved_modifications(self):
        """
        Returns
        -------
        bool
            whether or not the process should continue
        """
        if self.unsaved_modifications:
            ret = QMessageBox.warning(self, "Warning", "The data has been modified.\nDo you want to save your changes?",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
            if ret == QMessageBox.Save:
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
    # for ideas (# IPython/core/magics/code.py -> CodeMagics -> load)
    def _load_script(self, filepath):
        assert qtconsole_available
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            self.eval_box.input_buffer = content
            self.ipython_cell_executed()
            self.recent_loaded_scripts.add(filepath)
        except Exception as e:
            QMessageBox.critical(self, "Error", "Cannot load script file {}:\n{}"
                                 .format(os.path.basename(filepath), e))

    def load_script(self, filepath=None):
        # %load add automatically the extension .py if not present in passed filename
        dialog = QDialog(self)
        layout = QGridLayout()
        dialog.setLayout(layout)

        # filepath
        browse_label = QLabel("Source")
        browse_combobox = QComboBox()
        browse_combobox.setEditable(True)
        browse_combobox.addItems(self.recent_loaded_scripts.files)
        browse_combobox.lineEdit().setPlaceholderText("filepath to the python source")
        browse_button = QPushButton("Browse")
        if isinstance(filepath, str):
            browse_combobox.setText(filepath)
        browse_filedialog = QFileDialog(self, filter="Python Script (*.py)")
        browse_filedialog.setFileMode(QFileDialog.ExistingFile)
        browse_button.clicked.connect(browse_filedialog.open)
        browse_filedialog.fileSelected.connect(browse_combobox.lineEdit().setText)
        layout.addWidget(browse_label, 0, 0)
        layout.addWidget(browse_combobox, 0, 1)
        layout.addWidget(browse_button, 0, 2)

        # # lines / symbols
        # group_box = QGroupBox()
        # group_box_layout = QGridLayout()
        # # all lines
        # radio_button_all_lines = QRadioButton("Load all file")
        # radio_button_all_lines.setChecked(True)
        # group_box_layout.addWidget(radio_button_all_lines, 0, 0)
        # # specific lines
        # radio_button_specific_lines = QRadioButton("Load specific lines")
        # radio_button_specific_lines.setToolTip("Selected (ranges of) lines to load must be separated with "
        #                                        "whitespaces.\nRanges could be specified as x..y (x-y) or in "
        #                                        "python-style x:y (x..(y-1)).")
        # lines_edit = QLineEdit()
        # lines_edit.setPlaceholderText("1 4..6 8")
        # lines_edit.setEnabled(False)
        # radio_button_specific_lines.toggled.connect(lines_edit.setEnabled)
        # group_box_layout.addWidget(radio_button_specific_lines, 1, 0)
        # group_box_layout.addWidget(lines_edit, 1, 1)
        # # specific symbols (variables, functions and classes)
        # radio_button_symbols = QRadioButton("Load symbols")
        # symbols_edit = QLineEdit()
        # symbols_edit.setPlaceholderText("variables or functions separated by commas")
        # symbols_edit.setEnabled(False)
        # radio_button_symbols.toggled.connect(symbols_edit.setEnabled)
        # group_box_layout.addWidget(radio_button_symbols, 2, 0)
        # group_box_layout.addWidget(symbols_edit, 2, 1)
        # # set layout
        # group_box.setLayout(group_box_layout)
        # layout.addWidget(group_box, 1, 0, 1, 3)

        # clear session
        clear_session_checkbox = QCheckBox("Clear session before to load")
        clear_session_checkbox.setChecked(False)
        layout.addWidget(clear_session_checkbox, 1, 0, 1, 3)

        # accept/reject
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(dialog.accept)
        bbox.rejected.connect(dialog.reject)
        layout.addWidget(bbox, 2, 0, 1, 3)

        # open dialog
        ret = dialog.exec_()
        if ret == QDialog.Accepted:
            filepath = browse_combobox.currentText()
            if clear_session_checkbox.isChecked():
                self._reset()
            self._load_script(filepath)

    def _save_script(self, filepath, lines, overwrite):
        # IPython/core/magics/code.py -> CodeMagics -> save
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
            self.recent_saved_scripts.add(filepath)
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
        browse_combobox = QComboBox()
        browse_combobox.setEditable(True)
        browse_combobox.addItems(self.recent_saved_scripts.files)
        browse_combobox.lineEdit().setPlaceholderText("destination file")
        browse_button = QPushButton("Browse")
        browse_filedialog = QFileDialog(self, filter="Python Script (*.py)")
        browse_button.clicked.connect(browse_filedialog.open)
        browse_filedialog.fileSelected.connect(browse_combobox.lineEdit().setText)
        layout.addWidget(browse_label, 0, 0)
        layout.addWidget(browse_combobox, 0, 1)
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
        group_box_layout.addWidget(radio_button_overwrite, 0, 0)
        # append to
        radio_button_append = QRadioButton("Append to file")
        radio_button_append.setChecked(True)
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
            filepath = browse_combobox.currentText()
            if filepath == '':
                QMessageBox.warning(self, "Warning", "No file provided")
            else:
                specific_lines = radio_button_specific_lines.isChecked()
                lines = lines_edit.text() if specific_lines else ''

                overwrite = radio_button_overwrite.isChecked()
                if overwrite and os.path.isfile(filepath):
                    ret = QMessageBox.warning(self, "Warning",
                                              "File `{}` exists. Are you sure to overwrite it?".format(filepath),
                                              QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
                    if ret == QMessageBox.Save:
                        self._save_script(filepath, lines, overwrite)
                else:
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
            self._push_data(session)
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

    def _save_data(self, filepath):
        try:
            session = Session({k: v for k, v in self.data.items() if self._display_in_grid(k, v)})
            session.save(filepath)
            self.set_current_file(filepath)
            self.edit_undo_stack.clear()
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


class ArrayEditor(AbstractEditor):
    """Array Editor Dialog"""

    name = "Array Editor"

    def __init__(self, parent=None):
        AbstractEditor.__init__(self, parent, editable=True)
        self.setup_menu_bar()

    def _setup_and_check(self, widget, data, title, readonly, minvalue=None, maxvalue=None):
        """Setup ArrayEditor"""

        if np.isscalar(data):
            readonly = True

        layout = QVBoxLayout()
        widget.setLayout(layout)

        self.data = data
        self.arraywidget = ArrayEditorWidget(self, data, readonly, minvalue=minvalue, maxvalue=maxvalue)
        self.arraywidget.dataChanged.connect(self.push_changes)
        self.arraywidget.model_data.dataChanged.connect(self.update_title)
        self.update_title()
        layout.addWidget(self.arraywidget)

    def update_title(self):
        self._update_title(None, self.data, '')

    def push_changes(self, changes):
        self.edit_undo_stack.push(EditCurrentArrayCommand(self, self.data, changes))
