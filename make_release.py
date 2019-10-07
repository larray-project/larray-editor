#!/usr/bin/python
# coding=utf-8
# Release script for Editor
# Licence: GPLv3
# Requires:
# * git
from __future__ import print_function, unicode_literals

import sys
import json
from os.path import abspath, dirname, join
from releaser import make_release, insert_step_func, echocall, doechocall, no
from releaser import update_feedstock
from releaser.make_release import steps_funcs as make_release_steps
from releaser.update_feedstock import steps_funcs as update_feedstock_steps
from releaser.utils import chdir, short


TMP_PATH = r"c:\tmp\editor_new_release"
TMP_PATH_CONDA = r"c:\tmp\editor_conda_new_release"
PACKAGE_NAME = "larray-editor"
SRC_CODE = "larray_editor"
SRC_DOC = join('doc', 'source')
CONDA_BUILD_ARGS = {'--user': 'larray-project'}

GITHUB_REP = "https://github.com/larray-project/larray-editor"
CONDA_FEEDSTOCK_REP = "https://github.com/larray-project/larray-editor-feedstock.git"
ONLINE_DOC = None


def update_version_in_json_used_by_menuinst(config):
    chdir(config['build_dir'])

    version = short(config['release_name'])
    package_name = config['package_name']
    menuinst_file = join('condarecipe', package_name, 'larray-editor.json')

    with open(menuinst_file) as mf:
        data = json.load(mf)
    menu_items = data['menu_items']
    for i, menu_item in enumerate(menu_items):
        if 'webbrowser' in menu_item:
            menu_items[i]['webbrowser'] = 'http://larray.readthedocs.io/en/{}'.format(version)
    with open(menuinst_file, mode='w') as mf:
        json.dump(data, mf, indent=4)

    # check and add to next commit
    print(echocall(['git', 'diff', menuinst_file]))
    if no('Do the version update changes look right?'):
        exit(1)
    doechocall('Adding', ['git', 'add', menuinst_file])


insert_step_func(update_version_in_json_used_by_menuinst, before='update_version')


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: {} [-c|--conda] release_name|dev [step|startstep:stopstep] [branch]".format(argv[0]))
        print("make release steps:", ', '.join(f.__name__ for f, _ in make_release_steps))
        print("update conda-forge feedstock steps:", ', '.join(f.__name__ for f, _ in update_feedstock_steps))
        sys.exit()

    if argv[1] == '-c' or argv[1] == '--conda':
        argv = argv[2:]
        update_feedstock(GITHUB_REP, CONDA_FEEDSTOCK_REP, SRC_CODE, *argv, tmp_dir=TMP_PATH_CONDA)
    else:
        local_repository = abspath(dirname(__file__))
        make_release(local_repository, PACKAGE_NAME, SRC_CODE, *argv[1:], src_documentation=SRC_DOC, tmp_dir=TMP_PATH,
                     conda_build_args=CONDA_BUILD_ARGS)
