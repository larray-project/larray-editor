import os
from setuptools import setup, find_packages


def readlocal(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


DISTNAME = 'larray-editor'
VERSION = '0.35'
AUTHOR = 'Gaetan de Menten, Geert Bryon, Johan Duyck, Alix Damman'
AUTHOR_EMAIL = 'gdementen@gmail.com'
DESCRIPTION = "Graphical User Interface for LArray library"
LONG_DESCRIPTION = readlocal("README.rst")
LONG_DESCRIPTION_CONTENT_TYPE = "text/x-rst"
SETUP_REQUIRES = []

# * jedi >=0.18 to workaround incompatibility between jedi <0.18 and
#   parso >=0.8 (see #220)
# * Technically, we should require larray >=0.35 because we need align_arrays
#   for compare(), but to make larray-editor releasable, we cannot depend on
#   larray X.Y when releasing larray-editor X.Y (see utils.py for more details)
#   TODO: require 0.35 for next larray-editor version and drop shim in utils.py
# * Pandas is required directly for a silly reason (to support converting
#   pandas dataframes to arrays before comparing them). We could make it an
#   optional dependency by lazily importing it but but since it is also
#   indirectly required via larray, it does not really matter.
# * we do not actually require PyQt6 but rather either PyQt5, PyQt6 or PySide6
#   but I do not know how to specify this
# * we also have optional dependencies (but I don't know how to specify them):
#   - 'xlwings' for the "Copy to Excel" context-menu action
#   - 'tables' (PyTables) to load the example datasets from larray
INSTALL_REQUIRES = ['jedi >=0.18', 'larray >=0.32', 'matplotlib', 'numpy',
                    'pandas', 'PyQt6', 'qtpy']
TESTS_REQUIRE = ['pytest']

LICENSE = 'GPLv3'
URL = 'https://github.com/larray-project/larray-editor'
PACKAGE_DATA = {'larray_editor': ['images/*']}
ENTRY_POINTS = {'gui_scripts': ['larray-editor = larray_editor.start:main']}

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Programming Language :: Python :: 3.13',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries',
]

setup(
    name=DISTNAME,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    url=URL,
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    entry_points=ENTRY_POINTS,
)
