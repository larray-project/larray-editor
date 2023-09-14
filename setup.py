import os
from setuptools import setup, find_packages


def readlocal(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


DISTNAME = 'larray-editor'
VERSION = '0.34.1'
AUTHOR = 'Gaetan de Menten, Geert Bryon, Johan Duyck, Alix Damman'
AUTHOR_EMAIL = 'gdementen@gmail.com'
DESCRIPTION = "Graphical User Interface for LArray library"
LONG_DESCRIPTION = readlocal("README.rst")
LONG_DESCRIPTION_CONTENT_TYPE = "text/x-rst"
SETUP_REQUIRES = []
# pyqt cannot be installed via pypi. Dependencies (pyqt, qtpy and matplotlib) moved to conda recipe
# requires larray >= 0.32 because of the LArray -> Array rename
# TODO: add qtpy as dependency and mention pyqt or pyside
#       when using pyqt, we require at least pyqt >= 4.6 (for API v2)
# jedi >=0.18 to workaround incompatibility between jedi <0.18 and parso >=0.8 (see #220)
INSTALL_REQUIRES = ['larray >=0.32', 'jedi >=0.18']
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
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
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
