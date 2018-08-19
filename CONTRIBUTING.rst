.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Report Bugs
-----------

Report bugs at https://gitlab.pik-potsdam.de/swim/swimpy/issues

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.


Submit a merge request
----------------------

Ready to contribute? Here's how to set up `swimpy` for local development.

1. Fork the `swimpy` repository on Gitlab.
2. Clone your fork locally::

    $ git clone git@gitlab.pik-potsdam.de:your_name_here/swimpy.git

3. Install your local copy into a virtualenv as described in
   :ref:`installation:Setup a Python environment` and
   :ref:`installation:Install from source`.

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. Commit your changes and push your branch to GitLab::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

6. Submit a pull request through the GitLab website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The tests in ``swimpy/tests`` should pass without error in Python 2.7 and
   the latest stable Python 3 release. Consider adding tests for the additional
   code as well.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring. If a module is
   added, a new file needs to be added to ``docs/modules`` and listed in
   ``docs/modules.rst``.
