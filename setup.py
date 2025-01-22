#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import os
import re
import codecs
from setuptools import setup, find_packages

requirements = [
    "pandas>=0.23.4",
    "django>=1.11.20, >=3.0",
    "parse>=1.9, <2.0",
    "matplotlib>=2.2.3",
    "model-manager>=0.8",
    "f90nml>=1.4",
    "evoalgos>=1.0"
]


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst') as changelog_file:
    changelog = changelog_file.read()

with open('requirements.txt') as reqd_file:
    requirements_dev = reqd_file.read().split()


def package_files(dir):
    return [os.path.join(p, f) for (p, d, n) in os.walk(dir) for f in n]


def find_version(*file_paths):
    """Recommended way of getting the version without importing swimpy from
    https://packaging.python.org/guides/single-sourcing-package-version"""
    fp = os.path.join(os.path.abspath(os.path.dirname(__file__)), *file_paths)
    with codecs.open(fp, 'r') as fp:
        version_file = fp.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='swimpy',
    version=find_version("swimpy", "__init__.py"),
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
