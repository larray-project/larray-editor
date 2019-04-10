from __future__ import absolute_import, division, print_function

import sys
import traceback
from inspect import getframeinfo, stack
from collections import OrderedDict

from qtpy.QtWidgets import QApplication, QMainWindow
import larray as la

from larray_editor.editor import REOPEN_LAST_FILE, MappingEditor, ArrayEditor

__all__ = ['view', 'edit', 'compare', 'REOPEN_LAST_FILE']


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
    print_caller_info
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

    if not isinstance(obj, la.Session) and hasattr(obj, 'keys'):
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
    _app = QApplication.instance()
    if _app is None:
        _app = qapplication()
        parent = None
    else:
        parent = _app.activeWindow()

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

    if dlg.setup_and_check(args, names=names, title=title):
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
