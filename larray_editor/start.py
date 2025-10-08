import os
import sys

from larray_editor.api import _show_dialog, create_edit_dialog


def call_edit(obj):
    # we do not use edit() so that we can have display_caller_info=False
    _show_dialog("Viewer", create_edit_dialog, obj=obj,
                 display_caller_info=False, add_larray_functions=True)


def main():
    args = sys.argv[1:]
    if len(args) > 1:
        print(f"Usage: {sys.argv[0]} [file_path]")
        sys.exit()
    elif len(args) == 1:
        obj = args[0]
    else:
        obj = {}
    if os.name == 'nt':
        stderr_path = os.path.join(os.getenv("TEMP"), "stderr-" + os.path.basename(sys.argv[0]))
        with open(os.devnull, "w") as out, open(stderr_path, "w") as err:
            sys.stdout = out
            sys.stderr = err
            call_edit(obj)
    else:
        call_edit(obj)


if __name__ == '__main__':
    main()
