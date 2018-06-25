# Adapted from
# https://github.com/kennethreitz/setup.py

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'porter'
DESCRIPTION = 'porter is a framework for exposing machine learning models via REST APIs.'
URL = 'https://github.com/CadentTech/porter'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.0'

# What packages are required for this module to be executed?
REQUIRED = [
    # package: version
    'Flask>=1.0.2',
    'numpy>=1.14.3',
    'pandas>=0.23.0',
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
    'keras-utils': ['keras==2.1.6', 'tensorflow==1.5.0'],
    'sklearn-utils': ['scikit-learn==0.19.1'],
    's3-utils': ['boto3==1.7.35'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------

# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=(
        'tests', 'examples'
    )),
    install_requires=REQUIRED,
    extras_require=EXTRAS_REQUIRED
)
