import os
import sys
from contextlib import redirect_stdout, redirect_stderr

from pathlib import Path

from larray_editor.api import (create_edit_dialog, _show_dialog,
                               display_exception)
from larray_editor.utils import common_ancestor


def protected_main():
    args = sys.argv[1:]

    # Note that we do not check for --help or --version (which would have been
    # nice) because we cannot output anything to the console anyway. This is
    # because our entry point uses project.gui-scripts (which internally use
    # pythonw.exe) and thus do not support printing to the console.
    if len(args) == 1 and args[0] in {"%1", "%*"}:
        # workaround for menuinst issue which requires %1 or %* to support file
        # associations but then uses the literal string "%*" in the desktop/
        # menu shortcuts
        args = []

    paths = [Path(p) for p in args]
    absolute_paths = [p.resolve() for p in paths]
    ancestor = common_ancestor(absolute_paths) if len(paths) >= 2 else None

    def get_varname(p, ancestor):
        import re
        if ancestor is not None and p.exists():
            rel_path = p.relative_to(ancestor)
            name = str(rel_path.with_suffix(''))
        else:
            name = p.stem
        if not name:
            return 'path'
        # Replace invalid characters with underscores
        name = re.sub(r'[^0-9a-zA-Z_]', '_', name)
        if not name or name[0].isdigit() or name[0] == '_':
            # using '_' makes the variable hidden
            name = "path" + name
        return name

    # This is an odd way to display errors, but it is the simplest way to do it
    # without printing to the console.
    obj = {
        get_varname(abspath, ancestor):
            abspath if abspath.exists()
                    else [f"'{p}' is not a valid file or directory."]
        for p, abspath in zip(paths, absolute_paths)
    }

    # we do not use edit() so that we can have display_caller_info=False
    _show_dialog("Viewer",
                 create_edit_dialog,
                 obj=obj,
                 display_caller_info=False,
                 add_larray_functions=True)


def main():
    if os.name == 'nt':
        arg0 = os.path.basename(sys.argv[0])
        tmp_dir = os.getenv("TEMP")
        stderr_path = os.path.join(tmp_dir, f"{arg0}-stderr.log")
        with open(os.devnull, "w") as out, open(stderr_path, "w") as err:
            with redirect_stdout(out), redirect_stderr(err):
                # Cannot use install_except_hook() nor rely on the except
                # hook set within _show_dialog(), because by the time
                # the except hook is called by an unhandled exception
                # the redirected stderr is already closed by the context
                # manager and nothing is logged.
                try:
                    protected_main()
                # This purposefully does not catch/logs KeyboardInterrupt or
                # SystemExit exceptions
                except Exception as e:
                    display_exception(e)
                    return 1
    else:
        protected_main()
    return 0


if __name__ == '__main__':
    main()
