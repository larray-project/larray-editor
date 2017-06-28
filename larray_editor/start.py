from __future__ import absolute_import, division, print_function

import os
import sys
from larray import edit

def main():
    if os.name == 'nt':
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.path.join(os.getenv("TEMP"), "stderr-" + os.path.basename(sys.argv[0])), "w")
    edit()
