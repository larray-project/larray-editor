import importlib
import io
import logging
import os
import re
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Union

# Python3.8 switched from a Selector to a Proactor based event loop for asyncio but they do not offer the same
# features, which breaks Tornado and all projects depending on it, including Jupyter consoles
# ref: https://github.com/larray-project/larray-editor/issues/208
if sys.platform.startswith("win") and sys.version_info >= (3, 8):
    import asyncio

    try:
        from asyncio import WindowsProactorEventLoopPolicy, WindowsSelectorEventLoopPolicy
    except ImportError:
        # not affected
        pass
    else:
        if type(asyncio.get_event_loop_policy()) is WindowsProactorEventLoopPolicy:
            asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

import matplotlib
# explicitly request Qt backend (fixes #278)
matplotlib.use('QtAgg')
import matplotlib.axes
import numpy as np
import pandas as pd

import larray as la

from qtpy.QtCore import Qt, QUrl, QSettings
from qtpy.QtGui import QDesktopServices, QKeySequence
from qtpy.QtWidgets import (QMainWindow, QWidget, QListWidget, QListWidgetItem, QSplitter, QFileDialog, QPushButton,
                            QDialogButtonBox, QShortcut, QVBoxLayout, QGridLayout, QLineEdit,
                            QCheckBox, QComboBox, QMessageBox, QDialog,
                            QInputDialog, QLabel, QGroupBox, QRadioButton,
                            QTabWidget)

try:
    from qtpy.QtWidgets import QUndoStack
except ImportError:
    # PySide6 provides QUndoStack in QtGui
    # unsure qtpy has been fixed yet (see https://github.com/spyder-ide/qtpy/pull/366 for the fix for QUndoCommand)
    from qtpy.QtGui import QUndoStack

from larray_editor.traceback_tools import StackSummary
from larray_editor.utils import (_,
                                 create_action,
                                 show_figure,
                                 ima,
                                 commonpath,
                                 DEPENDENCIES,
                                 get_versions,
                                 get_documentation_url,
                                 URLS,
                                 RecentlyUsedList,
                                 logger)
from larray_editor.arraywidget import ArrayEditorWidget
from larray_editor import arrayadapter
from larray_editor.commands import EditSessionArrayCommand, EditCurrentArrayCommand
from larray_editor.sql import SQLWidget

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

ASSIGNMENT_PATTERN = re.compile(r'[^\[\]]+[^=]=[^=].+')
SUBSET_UPDATE_PATTERN = re.compile(r'(\w+)'
                                   r'(\.i|\.iflat|\.points|\.ipoints)?'
                                   r'\[.+\]\s*'
                                   r'([-+*/%&|^><]|//|\*\*|>>|<<)?'
                                   r'=\s*[^=].*')
HISTORY_VARS_PATTERN = re.compile(r'_i?\d+')
CAN_CONVERT_TO_LARRAY = (la.Array, np.ndarray, pd.DataFrame)

opened_secondary_windows = []

# TODO: remember its size
#       like MappingEditor via self.set_window_size_and_geometry()
class EditorWindow(QWidget):
    default_width = 800
    default_height = 600
    # This is more or less the minimum space required to display a 1D array
    minimum_width = 300
    minimum_height = 180
    name = "Editor"

    def __init__(self, data, title=None, readonly=False):
        # for QWidget to act as a window, parent must be None
        super().__init__(parent=None)
        layout = QVBoxLayout()
        self.setLayout(layout)
        array_widget = ArrayEditorWidget(self, data=data, readonly=readonly)
        self.array_widget = array_widget
        layout.addWidget(array_widget)

        icon = ima.icon('larray')
        if icon is not None:
            self.setWindowIcon(icon)

        if title is None:
            title = self.name
        self.setWindowTitle(title)
        # TODO: somehow determine better width
        self.resize(self.default_width, self.default_height)

    def closeEvent(self, event):
        logger.debug('EditorWindow.closeEvent()')
        if self in opened_secondary_windows:
            opened_secondary_windows.remove(self)
        super().closeEvent(event)
        self.array_widget.close()


class AbstractEditorWindow(QMainWindow):
    """Abstract Editor Window"""

    name = "Editor"
    editable = False
    file_menu = False
    help_menu = False
    default_width = 1000
    default_height = 600
    # This is more or less the minimum space required to display a 1D array
    minimum_width = 300
    minimum_height = 180

    def __init__(self, title='', readonly=False, caller_info=None, parent=None):
        QMainWindow.__init__(self, parent)

        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)

        self.data = None
        if self.editable:
            self.edit_undo_stack = QUndoStack(self)

        self.settings_group_name = self.name.lower().replace(' ', '_')
        self.widgets_to_save_to_settings = {}

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
            caller_info = f'launched from file {caller_info.filename} at line {caller_info.lineno}'
            self.statusBar().addPermanentWidget(QLabel(caller_info))
        # display welcome message
        self.statusBar().showMessage(f"Welcome to the {self.name}", 4000)

        # set central widget
        widget = QWidget()
        self.setCentralWidget(widget)

    def setup_menu_bar(self):
        """Setup menu bar"""
        menu_bar = self.menuBar()
        if self.file_menu:
            self._setup_file_menu(menu_bar)
        if self.editable:
            self._setup_edit_menu(menu_bar)
        if self.help_menu:
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
        # ============= #
        # DOCUMENTATION #
        # ============= #
        help_menu.addAction(create_action(self, _('Online &Documentation'), shortcut="Ctrl+H",
                                          triggered=self.open_documentation))
        help_menu.addAction(create_action(self, _('Online &Tutorial'), triggered=self.open_tutorial))
        help_menu.addAction(create_action(self, _('Online Objects and Functions (API) &Reference'),
                                          triggered=self.open_api_documentation))
        # ==================== #
        # ISSUES/GOOGLE GROUPS #
        # ==================== #
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
        # =============== #
        #       ABOUT     #
        # =============== #
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
            issue_template += f"* {package} {{{package}}}\n"
            for dep in DEPENDENCIES[package]:
                issue_template += f"* {dep} {{{dep}}}\n"
            issue_template = issue_template.format(**versions)

            url = QUrl(URLS[f'new_issue_{package}'])
            from qtpy.QtCore import QUrlQuery
            query = QUrlQuery()
            query.addQueryItem("body", quote(issue_template))
            url.setQuery(query)
            QDesktopServices.openUrl(url)

        return _report_issue

    def open_users_group(self):
        QDesktopServices.openUrl(QUrl(URLS['users_group']))

    def open_announce_group(self):
        QDesktopServices.openUrl(QUrl(URLS['announce_group']))

    def about(self):
        """About Editor"""
        kwargs = get_versions('editor')
        kwargs.update(URLS)
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
        for dep in DEPENDENCIES['editor']:
            message += f"<li>{dep} {kwargs[dep]}</li>\n"
        message += "</ul>"
        QMessageBox.about(self, _("About LArray Editor"), message.format(**kwargs))

    def _update_title(self, title, value, name):
        if title is None:
            title = []

        if value is not None:
            # TODO: the type-specific information added to the title should be
            #       computed by a method on the adapter
            #       (self.array_widget.data_adapter)
            if hasattr(value, 'dtype'):
                try:
                    dtype_str = f' [{value.dtype.name}]'
                except Exception:
                    dtype_str = ''
            else:
                dtype_str = ''

            if hasattr(value, 'shape'):
                def format_int(value: int):
                    if value >= 10_000:
                        return f'{value:_}'
                    else:
                        return str(value)

                if isinstance(value, la.Array):
                    shape = [f'{display_name} ({format_int(len(axis))})'
                             for display_name, axis in zip(value.axes.display_names, value.axes)]
                else:
                    try:
                        shape = [format_int(length) for length in value.shape]
                    except Exception:
                        shape = []
                shape_str = ' x '.join(shape)
            else:
                shape_str = ''

            # name + shape + dtype
            value_info = shape_str + dtype_str
            if name and value_info:
                title.append(name + ': ' + value_info)
            elif name:
                title.append(name)
            elif value_info:
                title.append(value_info)

        # extra info
        title.append(self._title)
        # set title
        self.setWindowTitle(' - '.join(title))

    def get_value(self):
        """Return modified array -- this is *not* a copy"""
        # It is import to avoid accessing Qt C++ object as it has probably
        # already been destroyed, due to the Qt.WA_DeleteOnClose attribute
        return self.data

    def save_widgets_state_and_geometry(self):
        settings = QSettings()
        settings.beginGroup(self.settings_group_name)
        settings.setValue('geometry', self.saveGeometry())
        settings.setValue('state', self.saveState())
        for widget_name, widget in self.widgets_to_save_to_settings.items():
            settings.beginGroup(f'widget/{widget_name}')
            if hasattr(widget, 'save_to_settings'):
                widget.save_to_settings(settings)
            elif hasattr(widget, 'saveState'):
                settings.setValue('state', widget.saveState())
            settings.endGroup()
        settings.endGroup()

    def restore_widgets_state_and_geometry(self):
        settings = QSettings()
        settings.beginGroup(self.settings_group_name)
        geometry = settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value('state')
        if state:
            self.restoreState(state)
        for widget_name, widget in self.widgets_to_save_to_settings.items():
            settings.beginGroup(f'widget/{widget_name}')
            if hasattr(widget, 'load_from_settings'):
                widget.load_from_settings(settings)
            elif hasattr(widget, 'restoreState'):
                widget_state = settings.value('state')
                if widget_state:
                    widget.restoreState(widget_state)
            settings.endGroup()
        settings.endGroup()
        return (geometry is not None) or (state is not None)

    def set_window_size_and_geometry(self):
        if not self.restore_widgets_state_and_geometry():
            self.resize(self.default_width, self.default_height)
        self.setMinimumSize(self.minimum_width, self.minimum_height)

    def update_title(self):
        raise NotImplementedError()


def void_formatter(obj, p, cycle):
    """
    p: PrettyPrinter
        has a .text() method to output text.
    cycle: bool
        Indicates whether the object is part of a reference cycle.
    """
    adapter_creator = arrayadapter.get_adapter_creator(obj)
    if isinstance(adapter_creator, str):
        # the string is an error message => we cannot handle that object
        #                                => use normal formatting
        # we can get in this case if we registered a void_formatter for a type
        # (such as Sequence) for which we handle some instances of the type
        # but not all
        p.text(repr(obj))
    else:
        # we already display the object in the grid
        #    => do not print it in the console
        return


class MappingEditorWindow(AbstractEditorWindow):
    """Session Editor Dialog"""

    name = "Session Editor"
    editable = True
    file_menu = True
    help_menu = True

    def __init__(self, data, title='', readonly=False, caller_info=None,
                 parent=None, stack_pos=None, add_larray_functions=False,
                 python_console=True, sql_console=None):
        AbstractEditorWindow.__init__(self, title=title, readonly=readonly, caller_info=caller_info,
                                      parent=parent)

        if sql_console is None:
            # This was meant to test whether users actually imported polars
            # in their script instead of just testing whether polars is present
            # in their environment but, in practice, this currently only does
            # the later because: larray_editor unconditionally imports larray
            # which imports xlwings when available, which imports polars when
            # available.
            sql_console = 'polars' in sys.modules
            logger.debug("polars module is present, enabling SQL console")
        elif sql_console:
            if importlib.util.find_spec('polars') is None:
                raise RuntimeError("SQL console is not available because "
                                   "the 'polars' module is not available")
        self.current_file = None
        self.current_array = None
        self.current_expr_text = None

        self.expressions = {}
        self.ipython_kernel = None
        self._unsaved_modifications = False

        # to handle recently opened data/script files
        self.recent_data_files = RecentlyUsedList("recentFileList", self, self.open_recent_file)
        self.recent_saved_scripts = RecentlyUsedList("recentSavedScriptList")
        self.recent_loaded_scripts = RecentlyUsedList("recentLoadedScriptList")

        self.setup_menu_bar()

        widget = self.centralWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        self._listwidget = QListWidget(self)
        # this is a bit more reliable than currentItemChanged which is not emitted when no item was selected before
        self._listwidget.itemSelectionChanged.connect(self.on_selection_changed)
        self._listwidget.itemDoubleClicked.connect(self.display_item_in_new_window)
        self._listwidget.setMinimumWidth(45)

        del_item_shortcut = QShortcut(QKeySequence(Qt.Key_Delete), self._listwidget)
        del_item_shortcut.activated.connect(self.delete_current_item)

        self.data = la.Session()
        self.array_widget = ArrayEditorWidget(self, readonly=readonly)
        self.array_widget.dataChanged.connect(self.push_changes)
        # FIXME: this is currently broken as it fires for each scroll
        #        we either need to fix model_data.dataChanged (but that might
        #        be needed for display) or find another way to add a star to
        #        the window title *only* when the user actually changed
        #        something
        # self.array_widget.model_data.dataChanged.connect(self.update_title)

        if sql_console:
            sql_widget = SQLWidget(self)
            self.widgets_to_save_to_settings['sql_console'] = sql_widget
        else:
            sql_widget = None
        self.sql_widget = sql_widget
        if python_console:
            if qtconsole_available:
                # silence a warning on Python 3.11 (see issue #263)
                if "PYDEVD_DISABLE_FILE_VALIDATION" not in os.environ:
                    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

                # Create an in-process kernel
                kernel_manager = QtInProcessKernelManager()
                kernel_manager.start_kernel(show_banner=False)
                kernel = kernel_manager.kernel
                self.ipython_kernel = kernel

                text_formatter = kernel.shell.display_formatter.formatters['text/plain']
                for type_ in arrayadapter.REGISTERED_ADAPTERS:
                    text_formatter.for_type(type_, void_formatter)

                kernel.shell.push({
                    '__editor__': self
                })

                if add_larray_functions:
                    kernel.shell.run_cell('from larray import *')
                    self.ipython_cell_executed()

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
                right_panel_widget.addWidget(self.array_widget)
                if sql_console:
                    tab_widget = QTabWidget(self)
                    tab_widget.addTab(self.eval_box, 'Python Console')
                    tab_widget.addTab(sql_widget, 'SQL Console')
                    right_panel_widget.addWidget(tab_widget)
                else:
                    right_panel_widget.addWidget(self.eval_box)

                right_panel_widget.setSizes([90, 10])
                self.widgets_to_save_to_settings['right_panel_widget'] = right_panel_widget
            else:
                # cannot easily use a QTextEdit because it has no returnPressed signal
                self.eval_box = QLineEdit()
                self.eval_box.returnPressed.connect(self.line_edit_update)

                right_panel_layout = QVBoxLayout()
                right_panel_layout.addWidget(self.array_widget)
                right_panel_layout.addWidget(self.eval_box)

                # you cant add a layout directly in a splitter, so we have to wrap
                # it in a widget
                right_panel_widget = QWidget()
                right_panel_widget.setLayout(right_panel_layout)
        elif sql_console:
            right_panel_widget = QSplitter(Qt.Vertical)
            right_panel_widget.addWidget(self.array_widget)
            right_panel_widget.addWidget(sql_widget)

            right_panel_widget.setSizes([90, 10])
            self.widgets_to_save_to_settings['right_panel_widget'] = right_panel_widget

        main_splitter = QSplitter(Qt.Horizontal)
        debug = isinstance(data, StackSummary)
        if debug:
            self._stack_frame_widget = QListWidget(self)
            stack_frame_widget = self._stack_frame_widget
            stack_frame_widget.itemSelectionChanged.connect(self.on_stack_frame_changed)
            stack_frame_widget.setMinimumWidth(60)

            for frame_summary in data:
                funcname = frame_summary.name
                filename = os.path.basename(frame_summary.filename)
                listitem = QListWidgetItem(stack_frame_widget)
                listitem.setText(f"{funcname}, {filename}:{frame_summary.lineno}")
                # we store the frame summary object in the user data of the list
                listitem.setData(Qt.UserRole, frame_summary)
                listitem.setToolTip(frame_summary.line)
            row = stack_pos if stack_pos is not None else len(data) - 1
            stack_frame_widget.setCurrentRow(row)

            left_panel_widget = QSplitter(Qt.Vertical)
            left_panel_widget.addWidget(self._listwidget)
            left_panel_widget.addWidget(stack_frame_widget)
            left_panel_widget.setSizes([500, 200])
            data = self.data
        else:
            left_panel_widget = self._listwidget
        main_splitter.addWidget(left_panel_widget)
        main_splitter.addWidget(right_panel_widget)
        main_splitter.setSizes([180, 620])
        main_splitter.setCollapsible(1, False)
        self.widgets_to_save_to_settings['main_splitter'] = main_splitter

        layout.addWidget(main_splitter)

        # check if reopen last opened file
        if data is REOPEN_LAST_FILE:
            if len(self.recent_data_files.files) > 0:
                data = self.recent_data_file.files[0]
            else:
                data = la.Session()

        # load file if any
        if isinstance(data, (str, Path)):
            if os.path.isfile(data):
                self._open_file(data)
            else:
                QMessageBox.critical(self, "Error", f"File {data} could not be found")
                self.new()
        elif not debug:
            self._push_data(data)

        self.set_window_size_and_geometry()

    def _push_data(self, data):
        self.data = data if isinstance(data, la.Session) else la.Session(data)
        if self.ipython_kernel is not None:
            # Avoid displaying objects we handle in IPython console.

            # Sadly, we cannot do this for all objects we support without
            # trying to import all the modules we support (which is clearly not
            # desirable), because IPython has 3 limitations.
            # 1) Its support for "string types" requires
            #    specifying the exact submodule a type is at (for example:
            #    pandas.core.frame.DataFrame instead of pandas.DataFrame).
            #    I do not think this is a maintainable approach for us (that is
            #    why the registering adapters using "string types" does not
            #    require that) so we use real/concrete types instead.

            # 2) It only supports *exact* types, not subclasses, so we cannot
            #    just register a custom formatter for "object" and be done
            #    with it.

            # 3) We cannot do this "just in time" by doing it in response
            #    to either ipython_widget executed or executing signals which
            #    both happen too late (the value is already displayed by the
            #    time those signals are fired)

            # The combination of the above limitations mean that types
            # imported via the console will NOT use the void_formatter :(.
            text_formatter = self.ipython_kernel.shell.display_formatter.formatters['text/plain']
            unique_types = {type(v) for v in self.data.values()}
            for obj_type in unique_types:
                adapter_creator = arrayadapter.get_adapter_creator_for_type(obj_type)
                if adapter_creator is None:
                    # if None, it means we do not handle that type at all
                    # => do not touch its ipython formatter
                    continue

                # Otherwise, we know the type is at least partially handled
                # (at least some instances are displayed) so we register our
                # void formatter and rely on it to fallback to repr() if
                # a particular instance of a type is not handled.
                try:
                    current_formatter = text_formatter.for_type(obj_type)
                except KeyError:
                    current_formatter = None
                if current_formatter is not void_formatter:
                    logger.debug(f"applying void_formatter for {obj_type}")
                    text_formatter.for_type(obj_type, void_formatter)
            self.ipython_kernel.shell.push(dict(self.data.items()))
        var_names = [k for k, v in self.data.items() if self._display_in_varlist(k, v)]
        self.add_list_items(var_names)
        self._listwidget.setCurrentRow(0)
        if self.sql_widget is not None:
            self.sql_widget.update_completer_options(self.data)

    def on_stack_frame_changed(self):
        selected = self._stack_frame_widget.selectedItems()
        if selected:
            assert len(selected) == 1
            selected_item = selected[0]
            assert isinstance(selected_item, QListWidgetItem)

            frame_summary = selected_item.data(Qt.UserRole)
            frame_globals, frame_locals = frame_summary.globals, frame_summary.locals
            data = {k: frame_globals[k] for k in sorted(frame_globals.keys())}
            data.update({k: frame_locals[k] for k in sorted(frame_locals.keys())})

            # CHECK:
            # * This clears the undo/redo stack, which is safer but is not ideal.
            #   When inspecting, for all frames except the last one the editor should be readonly (we should allow
            #   creating new temporary variables but not change existing ones).
            # * Does changing the last frame values has any effect after quitting the editor?
            #   It would be nice if we could do that (possibly with a warning when quitting the debug window)
            self._reset()
            self._push_data(data)

    def _reset(self):
        self.data = la.Session()
        self._listwidget.clear()
        self.current_array = None
        self.current_expr_text = None
        self.edit_undo_stack.clear()
        if self.ipython_kernel is not None:
            self.ipython_kernel.shell.reset()
            self.ipython_cell_executed()
        else:
            self.eval_box.setText('None')
            self.line_edit_update()

    def _setup_file_menu(self, menu_bar):
        file_menu = menu_bar.addMenu('&File')
        # ============= #
        #      NEW      #
        # ============= #
        file_menu.addAction(create_action(self, _('&New'),
                                          shortcut="Ctrl+N",
                                          triggered=self.new))
        file_menu.addSeparator()
        # ============= #
        #     DATA      #
        # ============= #
        file_menu.addSeparator()
        open_tip = _('Load session from file')
        file_menu.addAction(create_action(self, _('&Open Data'),
                                          shortcut="Ctrl+O",
                                          triggered=self.open_data,
                                          statustip=open_tip))
        save_tip = _('Save all arrays as a session in a file')
        file_menu.addAction(create_action(self, _('&Save Data'),
                                          shortcut="Ctrl+S",
                                          triggered=self.save_data,
                                          statustip=save_tip))
        file_menu.addAction(create_action(self, _('Save Data &As'),
                                          triggered=self.save_data_as,
                                          statustip=save_tip))
        recent_files_menu = file_menu.addMenu("Open &Recent Data")
        for action in self.recent_data_files.actions:
            recent_files_menu.addAction(action)
        recent_files_menu.addSeparator()
        recent_files_menu.addAction(create_action(self, _('&Clear List'),
                                                  triggered=self.recent_data_files.clear))
        # ============= #
        #    EXAMPLES   #
        # ============= #
        file_menu.addSeparator()
        file_menu.addAction(create_action(self, _('&Load Example Dataset'),
                                          triggered=self.load_example))
        # ============= #
        #    EXPLORER   #
        # ============= #
        file_menu.addSeparator()
        file_menu.addAction(create_action(self, _('Open File &Explorer'),
                                          triggered=self.open_explorer))
        # ============= #
        #    SCRIPTS    #
        # ============= #
        if qtconsole_available:
            file_menu.addSeparator()
            file_menu.addAction(create_action(self, _('&Load from Script'),
                                              shortcut="Ctrl+Shift+O",
                                              triggered=self.load_script,
                                              statustip=_('Load script from file')))
            file_menu.addAction(create_action(self, _('&Save Command History To Script'),
                                              shortcut="Ctrl+Shift+S",
                                              triggered=self.save_script,
                                              statustip=_('Save command history in a file')))

        # ============= #
        #     QUIT      #
        # ============= #
        file_menu.addSeparator()
        file_menu.addAction(create_action(self, _('&Quit'),
                                          shortcut="Ctrl+Q",
                                          triggered=self.close))

    def push_changes(self, changes):
        self.edit_undo_stack.push(EditSessionArrayCommand(self, self.current_expr_text, changes))

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
        if isinstance(value, la.Array):
            listitem.setToolTip(str(value.info))

    def add_list_items(self, names):
        for name in names:
            self.add_list_item(name)

    def delete_list_item(self, to_delete):
        deleted_items = self._listwidget.findItems(to_delete, Qt.MatchExactly)
        if len(deleted_items) == 1:
            deleted_item_idx = self._listwidget.row(deleted_items[0])
            self._listwidget.takeItem(deleted_item_idx)

    def display_item_in_new_window(self, list_item):
        assert isinstance(list_item, QListWidgetItem)
        varname = str(list_item.text())
        value = self.data[varname]
        self.new_editor_window(value, varname)

    def new_editor_window(self, data, title: str, readonly: bool=False):
        window = EditorWindow(data, title=title, readonly=readonly)
        window.show()
        # this is necessary so that the window does not disappear immediately
        opened_secondary_windows.append(window)

    def select_list_item(self, to_display):
        changed_items = self._listwidget.findItems(to_display, Qt.MatchExactly)
        assert len(changed_items) == 1, \
            f"len(changed_items) should be 1 but is {len(changed_items)}:\n{changed_items!r}"
        prev_selected = self._listwidget.selectedItems()
        assert len(prev_selected) <= 1
        # if the currently selected item (value) need to be refreshed (e.g it was modified)
        if prev_selected and prev_selected[0] == changed_items[0]:
            # we need to update the array widget explicitly
            self.set_current_array(self.data[to_display], to_display)
        else:
            self._listwidget.setCurrentItem(changed_items[0])

    def update_mapping_and_varlist(self, value):
        # XXX: use ordered set so that the order is non-random if the underlying container is ordered?
        keys_before = set(self.data.keys())
        keys_after = set(value.keys())
        # Contains both new and keys for which the object id changed (but not deleted keys nor inplace modified keys).
        # Inplace modified arrays should be already handled in ipython_cell_executed by the setitem_pattern.
        changed_keys = [k for k in keys_after if value[k] is not self.data.get(k)]

        # when a key is re-assigned, it can switch from being displayable to non-displayable or vice versa
        displayable_keys_before = {k for k in keys_before if self._display_in_varlist(k, self.data[k])}
        displayable_keys_after = {k for k in keys_after if self._display_in_varlist(k, value[k])}
        deleted_displayable_keys = displayable_keys_before - displayable_keys_after
        new_displayable_keys = displayable_keys_after - displayable_keys_before
        # this can contain more keys than new_displayble_keys (because of existing keys which changed value)
        changed_displayable_keys = [k for k in changed_keys if self._display_in_varlist(k, value[k])]

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

        # 4) update sql completer options if needed
        if self.sql_widget is not None and (new_displayable_keys or deleted_displayable_keys):
            self.sql_widget.update_completer_options(self.data)

        # 5) return variable to display, if any (if there are more than one,
        #    return first)
        return changed_displayable_keys[0] if changed_displayable_keys else None

    def delete_current_item(self):
        current_item = self._listwidget.currentItem()
        name = str(current_item.text())
        del self.data[name]
        if self.ipython_kernel is not None:
            self.ipython_kernel.shell.del_var(name)
        self.unsaved_modifications = True
        self._listwidget.takeItem(self._listwidget.row(current_item))

    def line_edit_update(self):
        import larray as la
        last_input = self.eval_box.text()
        if ASSIGNMENT_PATTERN.match(last_input):
            context = self.data._objects.copy()
            exec(last_input, la.__dict__, context)
            varname = self.update_mapping_and_varlist(context)
            if varname is not None:
                self.select_list_item(varname)
                self.expressions[varname] = last_input
        else:
            cur_output = eval(last_input, la.__dict__, self.data)
            self.view_expr(cur_output, last_input)

    def view_expr(self, array, expr_text):
        self._listwidget.clearSelection()
        self.set_current_array(array, expr_text)

    def _display_in_varlist(self, k, v):
        return (self._display_in_grid(v) and not k.startswith('__') and
                # This is ugly (and larray specific) but I did not find an
                # easy way to exclude that specific variable. I do not think
                # it should be in larray top level namespace anyway.
                k != 'EXAMPLE_EXCEL_TEMPLATES_DIR')

    def _display_in_grid(self, v):
        return not isinstance(arrayadapter.get_adapter_creator(v), str)

    def ipython_cell_executed(self):
        user_ns = self.ipython_kernel.shell.user_ns
        ip_keys = {'In', 'Out', '_', '__', '___', '__builtin__', '_dh', '_ih', '_oh', '_sh', '_i', '_ii', '_iii',
                   'exit', 'get_ipython', 'quit'}
        # '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__',
        clean_ns = {k: v for k, v in user_ns.items()
                    if k not in ip_keys and not HISTORY_VARS_PATTERN.match(k)}

        # user_ns['_i'] is not updated yet (refers to the -2 item)
        # 'In' and '_ih' point to the same object (but '_ih' is supposed to be
        # the non-overridden one)
        cur_input_num = len(user_ns['_ih']) - 1
        last_input = user_ns['_ih'][-1]

        # In case of multi-line input, only care about the last line. This is
        # not perfect as things like:
        #   arr3[1] = 42
        #   a = 1
        # will not be picked up. But the setitem thing cannot be done perfectly
        # anyway (any called function can modify any array), short of hashing
        # the content of all variables and checking which ones actually
        # changed, which would be too slow when working with large arrays.
        # At least this is predicatable.
        # last_input can be an empty string (e.g. when running ipython_cell_executed() manually)
        last_input_last_line = last_input.splitlines()[-1].strip() if last_input else ''

        # _oh and Out are supposed to be synonyms but "_oh" is supposed to be the non-overridden one.
        # It would be easier to use '_' instead but that refers to the last output, not the output of the
        # last command. Which means that if the last command did not produce any output, _ is not modified.
        cur_output = user_ns['_oh'].get(cur_input_num)
        setitem_pattern_match = SUBSET_UPDATE_PATTERN.match(last_input_last_line)
        # setitem
        if setitem_pattern_match is not None:
            varname = setitem_pattern_match.group(1)
        # simple variable
        elif last_input_last_line in clean_ns:
            varname = last_input_last_line
        # any other statement
        else:
            # any statement can contain a call to a function which adds or
            # removes globals.
            # This gives the name of the first changed array. Changed in this
            # context is either newly defined, or assigned to a new object
            # (arrays changed via setitem are *NOT* picked up here).
            varname = self.update_mapping_and_varlist(clean_ns)

        # Can we display the output?
        if (varname is not None and
                # this is necessary to avoid an error if a user ever does
                # setitem on an (i)python special variables. Those do not
                # concern us
                varname in clean_ns and
                # we prefer displaying cur_output via selecting a variable
                # in the list but want to avoid selecting a variable if
                # cur_output is also displayble but does not correspond
                (not self._display_in_grid(cur_output) or
                 clean_ns[varname] is cur_output) and
                self._display_in_varlist(varname, clean_ns[varname])):

            # For better or worse, _save_data() only saves "displayable data"
            # so changes to variables we cannot display do not concern us,
            # and this line should not be moved outside the if condition.
            if setitem_pattern_match is not None:
                self.unsaved_modifications = True

            # TODO: this completely refreshes the array, including detecting
            #       scientific & ndigits, which is not always what we want for
            #       setitem on the current array.
            self.select_list_item(varname)
        elif cur_output is not None:
            if self._display_in_grid(cur_output):
                self.view_expr(cur_output, last_input_last_line)

        # This should *NOT* be combined in the "elif cur_output is not None"
        # block above because this can happen in the first "if" branch too
        if cur_output is not None:
            if 'inline' not in matplotlib.get_backend():
                figure = self._get_figure(cur_output)
                if figure is not None:
                    show_figure(figure, title=last_input_last_line, parent=self)

    def _get_figure(self, cur_output):
        if isinstance(cur_output, matplotlib.figure.Figure):
            return cur_output

        if isinstance(cur_output, np.ndarray) and cur_output.size > 0:
            first_output = cur_output.flat[0]
        # we use a different path for sequences than for arrays to avoid copying potentially
        # big non-array sequences using np.ravel(). This code does not support nested sequences,
        # but I am already unsure supporting simple non-array sequences is useful.
        elif isinstance(cur_output, Sequence) and len(cur_output) > 0:
            first_output = cur_output[0]
        else:
            first_output = cur_output

        if isinstance(first_output, matplotlib.axes.Axes):
            return first_output.figure

        return None

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
        name = self.current_expr_text if self.current_expr_text is not None else ''
        unsaved_marker = '*' if self.unsaved_modifications else ''
        if self.current_file is not None:
            basename = os.path.basename(self.current_file)
            if os.path.isdir(self.current_file):
                assert not name.endswith('.csv')
                fname = os.path.join(basename, f'{name}.csv')
                name = ''
            else:
                fname = basename
        else:
            fname = '<new>'

        array = self.current_array
        title = [f'{unsaved_marker}{fname}']
        self._update_title(title, array, name)

    def set_current_array(self, array, expr_text):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("")
            clsname = self.__class__.__name__
            msg = f"{clsname}.set_current_array(<...>, {expr_text!r})"
            logger.debug(msg)
            logger.debug('=' * len(msg))

        # we should NOT check that "array is not self.current_array" because
        # this method is also called to refresh the widget value because of an
        # inplace setitem

        if self.sql_widget is not None:
            self.sql_widget.update_completer_options(self.data, selected=array)
        # FIXME: we should never store the current_array but current_adapter instead
        self.current_array = array
        self.array_widget.set_data(array)
        self.current_expr_text = expr_text
        self.update_title()

    def set_current_file(self, filepath: Union[str, Path]):
        if filepath is not None:
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
        logger.debug('MappingEditorWindow.closeEvent()')
        # as per the example in the Qt doc
        # (https://doc.qt.io/qt-5/qwidget.html#closeEvent), we should *NOT* call
        # the superclass closeEvent() method in this case because all it does is
        # "event.accept()" unconditionally which results in the application
        # being closed regardless of what the user chooses (see #202).
        if self._ask_to_save_if_unsaved_modifications():
            self.save_widgets_state_and_geometry()
            self.array_widget.close()
            event.accept()
        else:
            event.ignore()

    #########################################
    #               FILE MENU               #
    #########################################

    def new(self):
        if self._ask_to_save_if_unsaved_modifications():
            self._reset()
            self.array_widget.set_data(la.empty(0))
            self.set_current_file(None)
            self.unsaved_modifications = False
            self.statusBar().showMessage("Viewer has been reset", 4000)

    # ============================== #
    #  METHODS TO SAVE/LOAD SCRIPTS  #
    # ============================== #

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
            QMessageBox.critical(self, "Error", f"Cannot load script file {os.path.basename(filepath)}:\n{e}")

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

                def complete_slice(s):
                    if '-' not in s and ':' not in s:
                        return s
                    elif '-' in s and ':' in s:
                        raise ValueError('cannot have both .. (or -) and : in the same slice')
                    elif '-' in s:
                        sep = '-'
                    else:
                        assert ':' in s
                        sep = ':'

                    start, stop = s.split(sep)
                    if start == '':
                        start = 1
                    if stop == '':
                        stop = self.ipython_kernel.shell.execution_count
                        if sep == ':':
                            stop += 1
                    return f'{start}{sep}{stop}'

                lines = ' '.join(complete_slice(s) for s in lines.split(' '))
            else:
                lines = f'1-{self.ipython_kernel.shell.execution_count}'

            with io.StringIO() as tmp_out:
                with redirect_stdout(tmp_out):
                    self.ipython_kernel.shell.run_line_magic('save', f'{overwrite} "{filepath}" {lines}')
                stdout = tmp_out.getvalue()
            if 'commands were written to file' not in stdout:
                raise Exception(stdout)
            self.recent_saved_scripts.add(filepath)
            self.statusBar().showMessage(f"Command history was saved to {os.path.basename(filepath)}", 6000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot save history as {os.path.basename(filepath)}:\n{e}")

    # See http://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-save
    # for more details
    def save_script(self):
        if self.ipython_kernel.shell.execution_count == 1:
            QMessageBox.critical(self, "Error", "Cannot save an empty command history")
            return

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
                                               "Ranges can be specified either as x..y (x to y included) or in "
                                               "python-style x:y (x to y not included).")
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
            filepath = browse_combobox.currentText()
            if filepath == '':
                QMessageBox.warning(self, "Warning", "No file provided")
            else:
                specific_lines = radio_button_specific_lines.isChecked()
                lines = lines_edit.text() if specific_lines else ''

                overwrite = radio_button_overwrite.isChecked()
                if overwrite and os.path.isfile(filepath):
                    ret = QMessageBox.warning(self, "Warning",
                                              f"File `{filepath}` exists. Are you sure to overwrite it?",
                                              QMessageBox.Save | QMessageBox.Cancel)
                    if ret == QMessageBox.Save:
                        self._save_script(filepath, lines, overwrite)
                else:
                    self._save_script(filepath, lines, overwrite)

    # ============================= #
    #  METHODS TO SAVE/LOAD DATA    #
    # ============================= #
    def _open_file(self, filepath: Union[str, Path]):
        session = la.Session()
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
            self.statusBar().showMessage(f"Loaded: {display_name}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Something went wrong during load of file(s) {display_name}:\n{e}")

    def open_data(self):
        if self._ask_to_save_if_unsaved_modifications():
            filter = "All (*.xls *xlsx *.h5 *.csv);;Excel Files (*.xls *xlsx);;HDF Files (*.h5);;CSV Files (*.csv)"
            res = QFileDialog.getOpenFileNames(self, filter=filter)
            # Qt5+ returns a tuple (filepaths, '') instead of a string
            filepaths = res[0]
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
                    QMessageBox.warning(self, "Warning", f"File {filepath} could not be found")

    def _save_data(self, filepath):
        CAN_BE_SAVED = (la.Array, la.Axis, la.Group)
        in_var_list = {k: v for k, v in self.data.items()
                       if self._display_in_varlist(k, v)}
        if not in_var_list:
            QMessageBox.warning(self, "Warning", "Nothing to save")
            return

        to_save = {k: v for k, v in in_var_list.items()
                   if isinstance(v, CAN_BE_SAVED)}
        if not to_save:
            msg = ("Nothing can be saved because "
                   "all the currently loaded variables "
                   "are of types which are not supported for saving.")
            QMessageBox.warning(self, "Warning: unsavable objects", msg)
            return

        unsaveable = in_var_list.keys() - to_save.keys()
        if unsaveable:
            object_names = ', '.join(sorted(unsaveable))
            QMessageBox.warning(self, "Warning: unsavable objects",
                                "The following variables are of types which "
                                "are not supported for saving and will be "
                                f"ignored:\n\n{object_names}")
        session = la.Session(to_save)
        try:
            session.save(filepath)
            self.set_current_file(filepath)
            self.edit_undo_stack.clear()
            self.unsaved_modifications = False
            self.statusBar().showMessage(f"Arrays saved in file {filepath}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Something went wrong during save in file {filepath}:\n{e}")

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

    def open_explorer(self):
        self.new_editor_window(Path('.'), title="File Explorer", readonly=True)


class ArrayEditorWindow(AbstractEditorWindow):
    """Array Editor Dialog"""

    name = "Array Editor"
    editable = True
    file_menu = False
    help_menu = False

    def __init__(self, data, title='', readonly=False, caller_info=None, parent=None,
                 minvalue=None, maxvalue=None):
        AbstractEditorWindow.__init__(self, title=title, readonly=readonly, caller_info=caller_info,
                                      parent=parent)
        self.setup_menu_bar()

        widget = self.centralWidget()
        if np.isscalar(data):
            readonly = True

        layout = QVBoxLayout()
        widget.setLayout(layout)

        self.data = data
        self.array_widget = ArrayEditorWidget(self, data, readonly, minvalue=minvalue, maxvalue=maxvalue)
        self.array_widget.dataChanged.connect(self.push_changes)
        self.array_widget.model_data.dataChanged.connect(self.update_title)
        self.update_title()
        layout.addWidget(self.array_widget)
        self.set_window_size_and_geometry()

    def update_title(self):
        self._update_title(None, self.data, '')

    def push_changes(self, changes):
        self.edit_undo_stack.push(EditCurrentArrayCommand(self, self.data, changes))
