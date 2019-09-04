from __future__ import absolute_import, division, print_function

import collections
import os
import sys
import traceback
from inspect import getframeinfo
from collections import OrderedDict

from qtpy.QtWidgets import QApplication
import larray as la

from larray_editor.editor import REOPEN_LAST_FILE, MappingEditor, ArrayEditor

__all__ = ['view', 'edit', 'compare', 'REOPEN_LAST_FILE', 'run_editor_on_exception']


def qapplication():
    return QApplication(sys.argv)


def find_names(obj, depth=0):
    """Return all names an object is bound to.

    Parameters
    ----------
    obj : object
        the object to find names for.
    depth : int
        depth of call frame to inspect. 0 is where find_names was called,
        1 the caller of find_names, etc.

    Returns
    -------
    list of str
        all names obj is bound to, sorted alphabetically. Can be [] if we
        computed an array just to view it.
    """
    # noinspection PyProtectedMember
    l = sys._getframe(depth + 1).f_locals
    names = [k for k, v in l.items() if v is obj]
    if any(not name.startswith('_') for name in names):
        names = [name for name in names if not name.startswith('_')]
    return sorted(names)


def get_title(obj, depth=0, maxnames=3):
    """Return a title for an object (a combination of the names it is bound to).

    Parameters
    ----------
    obj : object
        the object to find a title for.
    depth : int
        depth of call frame to inspect. 0 is where get_title was called,
        1 the caller of get_title, etc.

    Returns
    -------
    str
        title for obj. This can be '' if we computed an array just to view it.
    """
    names = find_names(obj, depth=depth + 1)
    # names can be == []
    # eg. view(arr['M'])
    if len(names) > maxnames:
        names = names[:maxnames] + ['...']
    return ', '.join(names)


def edit(obj=None, title='', minvalue=None, maxvalue=None, readonly=False, depth=0, display_caller_info=True):
    """
    Opens a new editor window.

    Parameters
    ----------
    obj : np.ndarray, LArray, Session, dict, str or REOPEN_LAST_FILE, optional
        Object to visualize. If string, array(s) will be loaded from the file given as argument.
        Passing the constant REOPEN_LAST_FILE loads the last opened file.
        Defaults to the collection of all local variables where the function was called.
    title : str, optional
        Title for the current object. Defaults to the name of the first object found in the caller namespace which
        corresponds to `obj` (it will use a combination of the 3 first names if several names correspond to the same
        object).
    minvalue : scalar, optional
        Minimum value allowed.
    maxvalue : scalar, optional
        Maximum value allowed.
    readonly : bool, optional
        Whether or not editing array values is forbidden. Defaults to False.
    depth : int, optional
        Stack depth where to look for variables. Defaults to 0 (where this function was called).
    display_caller_info: bool, optional
        Whether or not to display the filename and line number where the Editor has been called.
        Defaults to True.

    Examples
    --------
    >>> a1 = ndtest(3)                                                                                 # doctest: +SKIP
    >>> a2 = ndtest(3) + 1                                                                             # doctest: +SKIP
    >>> # will open an editor with all the arrays available at this point
    >>> # (a1 and a2 in this case)
    >>> edit()                                                                                         # doctest: +SKIP
    >>> # will open an editor for a1 only
    >>> edit(a1)                                                                                       # doctest: +SKIP
    """
    install_except_hook()

    _app = QApplication.instance()
    if _app is None:
        _app = qapplication()
        _app.setOrganizationName("LArray")
        _app.setApplicationName("Viewer")
        parent = None
    else:
        parent = _app.activeWindow()

    caller_frame = sys._getframe(depth + 1)
    if display_caller_info:
        caller_info = getframeinfo(caller_frame)
    else:
        caller_info = None

    if obj is None:
        global_vars = caller_frame.f_globals
        local_vars = caller_frame.f_locals
        obj = OrderedDict()
        obj.update([(k, global_vars[k]) for k in sorted(global_vars.keys())])
        obj.update([(k, local_vars[k]) for k in sorted(local_vars.keys())])

    if not isinstance(obj, (la.Session, la.LArray)) and hasattr(obj, 'keys'):
        obj = la.Session(obj)

    if not title and obj is not REOPEN_LAST_FILE:
        title = get_title(obj, depth=depth + 1)

    if obj is REOPEN_LAST_FILE or isinstance(obj, (str, la.Session)):
        dlg = MappingEditor(parent)
        assert minvalue is None and maxvalue is None
        setup_ok = dlg.setup_and_check(obj, title=title, readonly=readonly, caller_info=caller_info)
    else:
        dlg = ArrayEditor(parent)
        setup_ok = dlg.setup_and_check(obj, title=title, readonly=readonly, minvalue=minvalue, maxvalue=maxvalue,
                                       caller_info=caller_info)

    if setup_ok:
        dlg.show()
        _app.exec_()

    restore_except_hook()


def view(obj=None, title='', depth=0, display_caller_info=True):
    """
    Opens a new viewer window. Arrays are loaded in readonly mode and their content cannot be modified.

    Parameters
    ----------
    obj : np.ndarray, LArray, Session, dict or str, optional
        Object to visualize. If string, array(s) will be loaded from the file given as argument.
        Defaults to the collection of all local variables where the function was called.
    title : str, optional
        Title for the current object. Defaults to the name of the first object found in the caller namespace which
        corresponds to `obj` (it will use a combination of the 3 first names if several names correspond to the same
        object).
    depth : int, optional
        Stack depth where to look for variables. Defaults to 0 (where this function was called).
    display_caller_info: bool, optional
        Whether or not to display the filename and line number where the Editor has been called.
        Defaults to True.

    Examples
    --------
    >>> a1 = ndtest(3)                                                                                 # doctest: +SKIP
    >>> a2 = ndtest(3) + 1                                                                             # doctest: +SKIP
    >>> # will open a viewer showing all the arrays available at this point
    >>> # (a1 and a2 in this case)
    >>> view()                                                                                         # doctest: +SKIP
    >>> # will open a viewer showing only a1
    >>> view(a1)                                                                                       # doctest: +SKIP
    """
    edit(obj, title=title, readonly=True, depth=depth + 1, display_caller_info=display_caller_info)


def compare(*args, **kwargs):
    """
    Opens a new comparator window, comparing arrays or sessions.

    Parameters
    ----------
    *args : LArrays or Sessions
        Arrays or sessions to compare.
    title : str, optional
        Title for the window. Defaults to ''.
    names : list of str, optional
        Names for arrays or sessions being compared. Defaults to the name of the first objects found in the caller
        namespace which correspond to the passed objects.
    depth : int, optional
        Stack depth where to look for variables. Defaults to 0 (where this function was called).
    display_caller_info: bool, optional
        Whether or not to display the filename and line number where the Editor has been called.
        Defaults to True.
    rtol : float or int, optional
        The relative tolerance parameter (see Notes). Defaults to 0.
    atol : float or int, optional
        The absolute tolerance parameter (see Notes). Defaults to 0.
    nans_equal : boolean, optional
        Whether or not to consider NaN values at the same positions in the two arrays as equal.
        By default, an array containing NaN values is never equal to another array, even if that other array
        also contains NaN values at the same positions. The reason is that a NaN value is different from
        *anything*, including itself. Defaults to True.

    Notes
    -----
    For finite values, the following equation is used to test whether two values are equal:

        absolute(array1 - array2) <= (atol + rtol * absolute(array2))


    Examples
    --------
    >>> a1 = ndtest(3)                                                                                 # doctest: +SKIP
    >>> a2 = ndtest(3) + 1                                                                             # doctest: +SKIP
    >>> compare(a1, a2, title='first comparison')                                                      # doctest: +SKIP
    >>> compare(a1 + 1, a2, title='second comparison', names=['a1+1', 'a2'])                           # doctest: +SKIP
    """
    install_except_hook()

    title = kwargs.pop('title', '')
    names = kwargs.pop('names', None)
    depth = kwargs.pop('depth', 0)
    display_caller_info = kwargs.pop('display_caller_info', True)

    _app = QApplication.instance()
    if _app is None:
        _app = qapplication()
        parent = None
    else:
        parent = _app.activeWindow()

    caller_frame = sys._getframe(depth + 1)
    if display_caller_info:
        caller_info = getframeinfo(caller_frame)
    else:
        caller_info = None

    if any(isinstance(a, la.Session) for a in args):
        from larray_editor.comparator import SessionComparator
        dlg = SessionComparator(parent)
        default_name = 'session'
    else:
        from larray_editor.comparator import ArrayComparator
        dlg = ArrayComparator(parent)
        default_name = 'array'

    def get_name(i, obj, depth=0):
        obj_names = find_names(obj, depth=depth + 1)
        return obj_names[0] if obj_names else '%s %d' % (default_name, i)

    if names is None:
        # depth + 2 because of the list comprehension
        names = [get_name(i, a, depth=depth + 2) for i, a in enumerate(args)]
    else:
        assert isinstance(names, list) and len(names) == len(args)

    if dlg.setup_and_check(args, names=names, title=title, caller_info=caller_info, **kwargs):
        dlg.show()
        _app.exec_()

    restore_except_hook()

_orig_except_hook = sys.excepthook


def _qt_except_hook(type, value, tback):
    # only print the exception and do *not* exit the program
    traceback.print_exception(type, value, tback)


def install_except_hook():
    sys.excepthook = _qt_except_hook


def restore_except_hook():
    sys.excepthook = _orig_except_hook


_orig_display_hook = sys.displayhook


def _qt_display_hook(value):
    if isinstance(value, la.LArray):
        view(value)
    else:
        _orig_display_hook(value)


def install_display_hook():
    sys.displayhook = _qt_display_hook


def restore_display_hook():
    sys.displayhook = _orig_display_hook


def _trace_code_file(tb):
    return os.path.normpath(tb.tb_frame.f_code.co_filename)


def _get_vars_from_frame(frame):
    frame_globals, frame_locals = frame.f_globals, frame.f_locals
    d = collections.OrderedDict()
    d.update([(k, frame_globals[k]) for k in sorted(frame_globals.keys())])
    d.update([(k, frame_locals[k]) for k in sorted(frame_locals.keys())])
    return d


def _get_debug_except_hook(root_path=None, usercode_traceback=True):
    try:
        main_file = os.path.abspath(sys.modules['__main__'].__file__)
    except AttributeError:
        main_file = sys.executable

    if root_path is None:
        root_path = os.path.dirname(main_file)

    def excepthook(type, value, tback):
        # first try to go as far as the main module because in some cases (e.g. when we run the file via a debugger),
        # the top of the traceback is not always the main module)
        current_tb = tback
        while current_tb.tb_next and _trace_code_file(current_tb) != main_file:
            current_tb = current_tb.tb_next

        main_tb = current_tb if _trace_code_file(current_tb) == main_file else tback

        if usercode_traceback:
            if main_tb != current_tb:
                print("Warning: couldn't find frame corresponding to user code, showing the full traceback "
                      "and inspect last frame instead (which might be in library code)",
                      file=sys.stderr)
                limit = None
            else:
                user_tb_length = 1
                # continue as long as the next tb is still in the current project
                while current_tb.tb_next and _trace_code_file(current_tb.tb_next).startswith(root_path):
                    current_tb = current_tb.tb_next
                    user_tb_length += 1
                limit = user_tb_length
        else:
            limit = None
        traceback.print_exception(type, value, main_tb, limit=limit)
        print("\nlaunching larray editor to debug...", file=sys.stderr)
        edit(_get_vars_from_frame(current_tb.tb_frame))

    return excepthook


def run_editor_on_exception(root_path=None, usercode_traceback=True):
    """
    Run the editor when an unhandled exception (a fatal error) happens.

    Parameters
    ----------
    root_path : str, optional
        Defaults to None (the directory of the main script).
    usercode_traceback : bool, optional
        Whether or not to show only the part of the traceback (error log) which corresponds to the user code.
        Otherwise, it will show the complete traceback, including code inside libraries. Defaults to True.

    Notes
    -----
    sets sys.excepthook
    """
    sys.excepthook = _get_debug_except_hook(root_path=root_path, usercode_traceback=usercode_traceback)
