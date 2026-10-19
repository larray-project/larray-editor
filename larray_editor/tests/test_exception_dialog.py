import os
from unittest import mock

import pytest

from larray_editor.api import qt_display_exception


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
