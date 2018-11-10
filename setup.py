#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
from setuptools import setup, find_packages
import swimpy

requirements = [
    "pandas>=0.20, <0.30.0",
    "django>=1.11, <2.0",
    "parse>=1.8, <2.0",
    "matplotlib>=2.0, <3.0",
    "model-manager>=0.3",
]


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst') as changelog_file:
    changelog = changelog_file.read()

with open('requirements.txt') as reqd_file:
    requirements_dev = reqd_file.read().split()


def package_files(dir):
    return [os.path.join(p, f) for (p, d, n) in os.walk(dir) for f in n]


setup(
    name='swimpy',
    version=swimpy.__version__,
    description="A python package to interact and test the ecohydrological model SWIM.",
    long_description=readme + '\n\n' + changelog,
    author="Michel Wortmann",
    author_email='wortmann@pik-potsdam.de',
    url='https://github.com/mwort/swimpy',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    data_files=package_files('swimpy/resources'),
    license="MIT license",
    zip_safe=False,
    keywords='swimpy',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    scripts=['swimpy/scripts/swimpy'],
    test_suite='tests',
    tests_require=requirements_dev,
    setup_requires=requirements,
)
