#!/bin/bash

# Ref: https://github.com/ContinuumIO/menuinst/wiki/Menus-in-Conda-Recipes
# See https://docs.conda.io/projects/conda-build/en/latest/user-guide/environment-variables.html
#     for a list of environment variables that are set during the build process.

#mkdir -p "$PREFIX/Menu"
#if [ $OSX_ARCH ]
#then
#    cp "$RECIPE_DIR/larray-editor.json" "$PREFIX/Menu"
#    cp "$SRC_DIR/larray_editor/images/larray.png" "$PREFIX/Menu"
#else
#    cp "$RECIPE_DIR/larray-editor.json" "$PREFIX/Menu"
#    cp "$SRC_DIR/larray_editor/images/larray.png" "$PREFIX/Menu"
#fi

"$PYTHON" -m pip install . --no-deps -vv || exit 1
