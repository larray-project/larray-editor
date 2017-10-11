from __future__ import print_function

import os
from setuptools import setup, find_packages

def readlocal(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


DISTNAME = 'larray-editor'
VERSION = '0.26'
AUTHOR = 'Gaetan de Menten, Geert Bryon, Johan Duyck, Alix Damman'
AUTHOR_EMAIL = 'gdementen@gmail.com'
DESCRIPTION = "Graphical User Interface for LArray library"
LONG_DESCRIPTION = readlocal("README.rst")
# pyqt cannot be installed via pypi. Dependencies (pyqt, qtpy and matplotlib) moved to conda recipe
INSTALL_REQUIRES = ['larray']
TESTS_REQUIRE = ['pytest']
SETUP_REQUIRES = ['pytest-runner']

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
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
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
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    setup_requires=SETUP_REQUIRES,
    url=URL,
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    entry_points=ENTRY_POINTS,
)
