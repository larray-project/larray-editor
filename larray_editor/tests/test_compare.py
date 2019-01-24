from __future__ import absolute_import, division, print_function

"""Array editor test"""

import logging
from larray import Session, where

from larray_editor.api import *
from larray_editor.utils import logger
from larray_editor.tests.test_data import *


logger.setLevel(logging.DEBUG)


compare(arr3, arr3 + 1.0)
compare(np.random.normal(0, 1, size=(10, 2)), np.random.normal(0, 1, size=(10, 2)))
compare(Session(arr4=arr4, arr3=arr3, data=data2),
        Session(arr4=arr4 + 1.0, arr3=arr3 * 2.0, data=data2 * 1.05))
# compare(Session(arr2=arr2, arr3=arr3),
#         Session(arr2=arr2 + 1.0, arr3=arr3 * 2.0))

arr1 = ndtest((3, 3))
arr2 = 2 * arr1
arr3 = where(arr1 % 2 == 0, arr1, -arr1)
compare(arr1, arr2, arr3)