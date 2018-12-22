#!/usr/bin/env python
# -*- coding: utf-8 -*
from setuptools import find_packages, setup

VERSION = '0.0.0.DEV0'

setup(
    name='livehistoryplot',
    version=VERSION,
    packages=find_packages(exclude=('tests', )),
    entry_points={'console_scripts': []},
    include_package_data=True,
    zip_safe=False,
    description='Live plot of Keras model history',
    author='Igor Gotlibovych',
    author_email='igor.gotlibovych@gmail.com',
    license='MIT',
    install_requires=[],
    extras_require={'matplotlib': ['matplotlib']},
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)