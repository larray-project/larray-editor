import os
import sys

from larray_editor import edit


def call_edit():
    edit(*sys.argv[1:], display_caller_info=False, add_larray_functions=True)


def main():
    if os.name == 'nt':
        stderr_path = os.path.join(os.getenv("TEMP"), "stderr-" + os.path.basename(sys.argv[0]))
        with open(os.devnull, "w") as out, open(stderr_path, "w") as err:
            sys.stdout = out
            sys.stderr = err
            call_edit()
    else:
        call_edit()


if __name__ == '__main__':
    main()
