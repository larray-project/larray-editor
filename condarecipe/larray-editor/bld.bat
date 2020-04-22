:: Ref: https://github.com/ContinuumIO/menuinst/wiki/Menus-in-Conda-Recipes
:: See: https://docs.conda.io/projects/conda-build/en/latest/user-guide/environment-variables.html
::      for a list of environment variables that are set during the build process.
:: e.g. %PREFIX% = (...)\Miniconda

if not exist "%PREFIX%\Menu" mkdir "%PREFIX%\Menu"
copy "%RECIPE_DIR%\larray-editor.json" "%PREFIX%\Menu"
copy "%SRC_DIR%\larray_editor\images\larray.ico" "%PREFIX%\Menu"
copy "%SRC_DIR%\larray_editor\images\larray-help.ico" "%PREFIX%\Menu"

"%PYTHON%" -m pip install . --no-deps -vv
if errorlevel 1 exit 1
