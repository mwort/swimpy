#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as req_file:
    requirements = req_file.read().split()

with open('requirements_dev.txt') as reqd_file:
    requirements_dev = reqd_file.read().split()


setup(
    name='swimpy',
    version='0.1.0',
    description="A python package to interact and test the ecohydrological model SWIM.",
    long_description=readme + '\n\n' + history,
    author="Michel Wortmann",
    author_email='wortmann@pik-potsdam.de',
    url='https://github.com/mwort/swimpy',
    packages=find_packages(include=['swimpy']),
    include_package_data=True,
    install_requires=requirements,
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
    test_suite='tests',
    tests_require=requirements_dev,
    setup_requires=requirements,
    dependency_links=[
        "git+https://github.com/mwort/modelmanager.git#egg=modelmanager"
    ]
)
