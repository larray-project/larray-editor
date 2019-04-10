from __future__ import absolute_import, division, print_function

import os
import sys
import inspect
from larray_editor import edit


def main():
    if os.name == 'nt':
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.path.join(os.getenv("TEMP"), "stderr-" + os.path.basename(sys.argv[0])), "w")
    try:
        pos_arg = inspect.getfullargspec(edit).args.index('display_caller_info')
        if len(sys.argv) > pos_arg + 1:
            sys.argv[pos_arg + 1] = False
            edit(*sys.argv[1:])
        else:
            edit(*sys.argv[1:], display_caller_info=False)
    except ValueError:
        edit(*sys.argv[1:])


if __name__ == '__main__':
    main()
