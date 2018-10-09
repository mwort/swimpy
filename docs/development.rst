***********
Development
***********

.. todoList::

.. include:: ../AUTHORS.rst
.. include:: ../CHANGELOG.rst
.. include:: ../CONTRIBUTING.rst


=======
Testing
=======

As well as SWIMpy's build-in testing suite for SWIM (`swimpy.tests`),
the package comes with tests for its own code in ``tests/``. They rely on the
*SWIM* repository test project and the *m.swim* GRASS modules test grass database.
Both these dependencies are git submodules in the ``dependencies/`` directory
and their tests need to be run before running SWIMpy's tests::

   git submodule update --init --recursive
   cd dependencies/SWIM/project
   make
   cd ../m.swim/tests
   make

To run the SWIMpy tests, install the version-frozen requirements
(``pip install -r requirements.txt``) and then by running ``make`` in the
``tests/`` directory. Individual tests can be run like this (e.g.)::

   $ python tests.py TestParameters.test_subcatch_parameters


=============
Documentation
=============

The documentation is produced by
`Sphinx <http://www.sphinx-doc.org/en/master/index.html>`_
and consists of the `reStructuredText <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
files and Jupyter Notebooks in ``docs/`` as well as the
*docstrings* included in the code. To build it, install the
``requirements_dev.txt`` packages, **run the tests** and execute::

    $ make docs

*docstrings* should be formatted according to the
`NumpyDoc standards <https://numpydoc.readthedocs.io/en/latest/>`_ and/or
according to the `docutil reStructuredText <http://docutils.sourceforge.net/docs/>`_.
There is an online quick renderer `here <http://rst.ninjs.org/>`_.


=========
Releasing
=========

A reminder for the maintainers on how to deploy:

1. Add entry to ``CHANGELOG.rst`` for major and minor.
2. Change version in ``swimpy.__init__.py``
   (`semanitc versioning <https://semver.org/>`_ major.minor.patch, no ``v``).
3. Commit changes (e.g. ``$ git commit -m "Release 0.1.8"``).
4. Tag commit with version number, e.g. `v1.2.0`
5. Push commits and tags: ``$ git push ; git push --tags``
6. Update docs: ``$ make servedocs``
