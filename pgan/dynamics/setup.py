#-*- coding: utf-8 -*-

from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup
import sys

def configuration(parent_package='', top_path=None):
    config = Configuration('dynamics', parent_package, top_path)
    config.add_subpackage('burgers')
    config.add_subpackage('navierstokes')
    config.add_subpackage('fluidsim')
    config.make_config_py()
    return config

if __name__ == '__main__':
    setup(configuration=configuration)

# end of file
