from __future__ import absolute_import, division, print_function

import sys
import traceback
from collections import OrderedDict

from qtpy.QtWidgets import QApplication, QMainWindow
import larray as la

from larray_editor.editor import REOPEN_LAST_FILE

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


def edit(obj=None, title='', minvalue=None, maxvalue=None, readonly=False, depth=0):
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

    if obj is None:
        caller_frame = sys._getframe(depth + 1)
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
        from larray_editor.editor import MappingEditor
        dlg = MappingEditor(parent)
    else:
        from larray_editor.editor import ArrayEditor
        dlg = ArrayEditor(parent)

    if dlg.setup_and_check(obj, title=title, readonly=readonly, minvalue=minvalue, maxvalue=maxvalue):
        if parent or isinstance(dlg, QMainWindow):
            dlg.show()
            _app.exec_()
        else:
            dlg.exec_()

    restore_except_hook()


def view(obj=None, title='', depth=0):
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
    edit(obj, title=title, readonly=True, depth=depth + 1)


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
        if parent:
            dlg.show()
        else:
            dlg.exec_()

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


if __name__ == "__main__":
    """Array editor test"""
    import numpy as np
    from larray import (Session, Axis, LArray, ndtest, ndrange, zeros, from_lists, union,
                        sin, cos, radians, maximum, sqrt)

    lipro = Axis(['P%02d' % i for i in range(1, 16)], 'lipro')
    age = Axis('age=0..115')
    sex = Axis('sex=M,F')

    vla = 'A11,A12,A13,A23,A24,A31,A32,A33,A34,A35,A36,A37,A38,A41,A42,A43,A44,A45,A46,A71,A72,A73'
    wal = 'A25,A51,A52,A53,A54,A55,A56,A57,A61,A62,A63,A64,A65,A81,A82,A83,A84,A85,A91,A92,A93'
    bru = 'A21'
    # list of strings
    belgium = union(vla, wal, bru)

    geo = Axis(belgium, 'geo')

    # data1 = np.arange(30).reshape(2, 15)
    # arr1 = la.LArray(data1, axes=(sex, lipro))
    # edit(arr1)

    # data2 = np.arange(116 * 44 * 2 * 15).reshape(116, 44, 2, 15) \
    #           .astype(float)
    # data2 = np.random.random(116 * 44 * 2 * 15).reshape(116, 44, 2, 15) \
    #           .astype(float)
    # data2 = (np.random.randint(10, size=(116, 44, 2, 15)) - 5) / 17
    # data2 = np.random.randint(10, size=(116, 44, 2, 15)) / 100 + 1567
    # data2 = np.random.normal(51000000, 10000000, size=(116, 44, 2, 15))
    data2 = np.random.normal(0, 1, size=(116, 44, 2, 15))
    arr2 = LArray(data2, axes=(age, geo, sex, lipro))
    # arr2 = ndrange([100, 100, 100, 100, 5])
    # arr2 = arr2['F', 'A11', 1]

    # view(arr2[0, 'A11', 'F', 'P01'])
    # view(arr1)
    # view(arr2[0, 'A11'])
    # edit(arr1)
    # print(arr2[0, 'A11', :, 'P01'])
    # edit(arr2.astype(int), minvalue=-99, maxvalue=55.123456)
    # edit(arr2.astype(int), minvalue=-99)
    # arr2.i[0, 0, 0, 0] = np.inf
    # arr2.i[0, 0, 1, 1] = -np.inf
    # arr2 = [0.0000111, 0.0000222]
    # arr2 = [0.00001, 0.00002]
    # edit(arr2, minvalue=-99, maxvalue=25.123456)
    # print(arr2[0, 'A11', :, 'P01'])

    # data2 = np.random.normal(0, 10.0, size=(5000, 20))
    # arr2 = LArray(data2, axes=(Axis(list(range(5000)), 'd0'),
    #                            Axis(list(range(20)), 'd1')))
    # edit(arr2)

    # view(['a', 'bb', 5599])
    # view(np.arange(12).reshape(2, 3, 2))
    # view([])

    data3 = np.random.normal(0, 1, size=(2, 15))
    arr3 = ndrange((30, sex))
    # data4 = np.random.normal(0, 1, size=(2, 15))
    # arr4 = LArray(data4, axes=(sex, lipro))

    # arr4 = arr3.copy()
    # arr4['F'] /= 2
    arr4 = arr3.min(sex)
    arr5 = arr3.max(sex)
    arr6 = arr3.mean(sex)

    # test isssue #35
    arr7 = from_lists([['a',                   1,                    2,                    3],
                       [ '', 1664780726569649730, -9196963249083393206, -7664327348053294350]])

    def make_circle(width=20, radius=9):
        x, y = Axis(width, 'x'), Axis(width, 'y')
        center = (width - 1) / 2
        return maximum(radius - sqrt((x - center) ** 2 + (y - center) ** 2), 0)

    def make_sphere(width=20, radius=9):
        x, y, z = Axis(width, 'x'), Axis(width, 'y'), Axis(width, 'z')
        center = (width - 1) / 2
        return maximum(radius - sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2), 0)

    def make_demo(width=20, ball_radius=5, path_radius=5, steps=30):
        x, y = Axis(width, 'x'), Axis(width, 'y')
        t = Axis(steps, 't')
        center = (width - 1) / 2
        ball_center_x = sin(radians(t * 360 / steps)) * path_radius + center
        ball_center_y = cos(radians(t * 360 / steps)) * path_radius + center
        return maximum(ball_radius - sqrt((x - ball_center_x) ** 2 + (y - ball_center_y) ** 2), 0).transpose(x, y)

    demo = make_demo(9, 2.5, 1.5)
    sphere = make_sphere(9, 4)
    extreme_array = LArray([-np.inf, -1, 0, np.nan, 1, np.inf])
    scalar = LArray(0)
    arr_empty = LArray([])
    arr_obj = ndtest((2, 3)).astype(object)
    arr_str = ndtest((2, 3)).astype(str)
    big = ndtest((1000, 1000, 500))

    # test autoresizing
    long_labels = zeros('a=a_long_label,another_long_label; b=this_is_a_label,this_is_another_one')
    long_axes_names = zeros('first_axis=a0,a1; second_axis=b0,b1')

    # compare(arr3, arr4, arr5, arr6)

    # view(stack((arr3, arr4), Axis('arrays=arr3,arr4')))
    # ses = Session(arr2=arr2, arr3=arr3, arr4=arr4, arr5=arr5, arr6=arr6, arr7=arr7, long_labels=long_labels,
    #                  long_axes_names=long_axes_names, data2=data2, data3=data3)

    # from larray.tests.common import abspath
    # file = abspath('test_session.xlsx')
    # ses.save(file)

    # import cProfile as profile
    # profile.runctx('edit(Session(arr2=arr2))', vars(), {},
    #                'c:\\tmp\\edit.profile')
    edit()
    # edit(ses)
    # edit(file)
    # edit('fake_path')
    # edit(REOPEN_LAST_FILE)

    # edit(arr2)
    # compare(arr3, arr3 + ndrange(arr3.axes))
    # compare(Session(arr4=arr4, arr3=arr3),
    #         Session(arr4=arr4 + 1.0, arr3=arr3 * 2.0))
    # compare(Session(arr2=arr2, arr3=arr3),
    #         Session(arr2=arr2 + 1.0, arr3=arr3 * 2.0))

    # s = local_arrays()
    # view(s)
    # print('HDF')
    # s.save('x.h5')
    # print('\nEXCEL')
    # s.save('x.xlsx')
    # print('\nCSV')
    # s.save('x_csv')
    # print('\n open HDF')
    # edit('x.h5')
    # print('\n open EXCEL')
    # edit('x.xlsx')
    # print('\n open CSV')
    # edit('x_csv')

    # compare(arr3, arr4, arr5, arr6)
