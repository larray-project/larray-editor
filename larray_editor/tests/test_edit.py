from __future__ import absolute_import, division, print_function

"""Array editor test"""

import logging

from larray_editor.api import *
from larray_editor.utils import logger
from larray_editor.tests.test_data import *


logger.setLevel(logging.DEBUG)


# import cProfile as profile
# profile.runctx('edit(Session(arr2=arr2))', vars(), {},
#                'c:\\tmp\\edit.profile')

edit()
# edit(ses)
# edit(file)
# edit('fake_path')
# edit(REOPEN_LAST_FILE)

edit(arr2)
