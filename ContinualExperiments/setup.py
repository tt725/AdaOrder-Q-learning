from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "Require Python 3.6 or greater."

setup(
    name='diffalgos',
    py_modules=['diffalgos'],
    version='0.0.1',
    install_requires=[
        'numpy',
        'joblib',
        'gym>=0.17.2'
    ],
    description="diffalgos algorithm PyTorch implementation",
    author="Tao Tan, Wentao Hu, Hong Xie, Defu Lian",
)
