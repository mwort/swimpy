.. highlight:: shell

============
Installation
============

Quick install
-------------

Before installing, it's recommended to setup a virtual python environment
(see below). With the environment activated and in a directory of your choice,
run the following to install (requires *git* and prompts twice for username/password):

.. code-block:: console

    $ pip install git+https://gitlab.pik-potsdam.de/wortmann/swimpy.git


Setup a Python environment
--------------------------

Before installing SWIMpy, it is highly advisable to first set up a virtual
environment. Install `virtualenv`_ first, unless you are using Anaconda (see below).
Then create a new environment like this:

.. code-block:: console

    $ virtualenv -p python2.7 swimpyenv

Then and every time before using SWIMpy, activate the environment like this:

.. code-block:: console

    $ source swimpyenv/bin/activate

Since GRASS does not fully support python3, it is recommended to use python2.7
but SWIMpy also supports python3.

To install any additional packages, install them like this inside the environment
(e.g. the interactive python console ``ipython`` is highly recommended):

.. code-block:: console

  $ pip install ipython


Anaconda
^^^^^^^^

Create a new conda environment and install the large dependencies like this:

.. code-block:: console

    $ conda create -y -n swimpyenv python=2.7 pip matplotlib pandas ipython

Then and every time before using SWIMpy (or add this line to your ``.bashrc``
file), activate the environment like this:

.. code-block:: console

    $ source activate swimpyenv

New packages can be installed like this:

.. code-block:: console

  $ pip install ipython


.. _virtualenv: https://virtualenv.pypa.io/en/stable/installation/


Install from source
-------------------

The SWIMpy source code can be downloaded from the `PIK GitLab repo`_.

You can either clone the public repository (makes updating easy):

.. code-block:: console

    $ git clone https://gitlab.pik-potsdam.de/wortmann/swimpy.git

Or download the project as `zip/tar`_.

Once you have a copy of the source code, you can install it with:

.. code-block:: console

    $ pip install -e swimpy/


``swimpy`` in the above is the downloaded directory/repository. Leave the ``-e``
out if you dont want to edit the code; you can then also remove the ``swimpy/``
directory.

.. _PIK GitLab repo: https://gitlab.pik-potsdam.de/wortmann/swimpy
.. _zip/tar: https://gitlab.pik-potsdam.de/wortmann/swimpy/repository/archive.zip?ref=master


Enable commandline autocompletion
---------------------------------

To autocomplete the swimpy commandline arguments and flags in a bash shell,
install the `argcomplete`_ package like this:

.. code-block:: console

    $ pip install argcomplete

Then add this line to your :code:`~/.bash_rc`/:code:`~/.bash_profile` file and open a new
shell::

    eval "$(register-python-argcomplete swimpy)"


.. _argcomplete: http://argcomplete.readthedocs.io
