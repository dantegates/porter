# Adapted from
# https://github.com/kennethreitz/setup.py

import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'porter'
DESCRIPTION = 'porter is a framework for exposing machine learning models via REST APIs.'
URL = 'https://github.com/CadentTech/porter'
REQUIRES_PYTHON = '>=3.6.0'

# What packages are required for this module to be executed?
REQUIRED = [
    # package: version
    'Flask>=1.0.2,<1.1.0',
    'numpy>=1.15.0',
    'pandas>=0.23.0',
    'joblib',
    'fastjsonschema>=2.14.4',
]

# These are packages required for non-essential functionality, e.g. loading
# keras models. These additional features can be installed with pip. Below is
# an example of how to install additional keras and s3 functionality.
# 
#    $ pip install porter[keras-utils,s3-utils]
#
# For more details see:
# http://peak.telecommunity.com/DevCenter/setuptools#declaring-extras-optional-features-with-their-own-dependencies
# and 
# https://github.com/seatgeek/fuzzywuzzy#installation
#
EXTRAS_REQUIRED = {
    'keras-utils': ['tensorflow>=1.4.0'],
    's3-utils': ['boto3>=1.7.65'],
}

EXTRAS_REQUIRED['all'] = [r for requirements in EXTRAS_REQUIRED.values() for r in requirements]


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------


here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(here, NAME, '__init__.py')) as f:
    exec(f.read(), about)
VERSION = about['__version__']


# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=(
        'tests',
    )),
    install_requires=REQUIRED,
    extras_require=EXTRAS_REQUIRED,
    include_package_data=True
)
