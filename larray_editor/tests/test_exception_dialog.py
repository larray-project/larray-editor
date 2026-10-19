import os
import sys
import types
from unittest import mock

import pytest

from larray_editor.api import _get_debug_except_hook, qt_display_exception


# TODO: temporary, remove this once these tests are known to work on CI, which runs without
#       display (they should, using the offscreen Qt platform, see conftest.py)
pytestmark = pytest.mark.skipif(bool(os.environ.get('CI')),
                                reason="Qt tests are temporarily skipped on CI")


def test_exception_message_looking_like_html():
    exception = TypeError("expected <class 'int'>")
    with mock.patch('larray_editor.api.QMessageBox') as message_box:
        qt_display_exception(exception)
    msg = message_box.call_args.args[2]
    assert "expected &lt;class 'int'&gt;" in msg


def test_debug_except_hook_root_path_with_slashes(tmp_path):
    # the exception happens in a library called by a user module called by the main script
    main_file = str(tmp_path / 'project' / 'main.py')
    user_file = str(tmp_path / 'project' / 'user.py')
    lib_file = str(tmp_path / 'lib.py')
    code = {}
    exec(compile('def fail():\n    raise ValueError()', lib_file, 'exec'), code)
    exec(compile('def user():\n    fail()', user_file, 'exec'), code)
    exec(compile('def main():\n    user()', main_file, 'exec'), code)
    try:
        code['main']()
    except ValueError as e:
        exception = e
    root_path = str(tmp_path / 'project').replace('\\', '/')
    main_module = types.SimpleNamespace(__file__=main_file)
    with mock.patch.dict(sys.modules, {'__main__': main_module}), \
            mock.patch('larray_editor.api._show_dialog') as show_dialog, \
            mock.patch('larray_editor.api.traceback.print_exception'):
        excepthook = _get_debug_except_hook(root_path=root_path)
        excepthook(type(exception), exception, exception.__traceback__)
    # the frames of main and user (not the frame of lib)
    stack = show_dialog.call_args.args[2]
    assert [frame.filename for frame in stack] == [main_file, user_file]
