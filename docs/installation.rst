.. highlight:: shell

============
Installation
============

Quick install
-------------

Before installing, it's recommended to setup a virtual python environment
(see below). With the environment activated and in a directory of your choice,
run the following to install (requires *git* and prompts for username/password):

.. code-block:: console

    $ git clone git+https://gitlab.pik-potsdam.de/wortmann/swimpy.git
    $ pip install swimpy


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
but SWIMpy also supporst python3.


Anaconda
^^^^^^^^

Create a new conda environment and install the large dependencies like this:

.. code-block:: console

    $ conda create -y -n swimpyenv python=2.7 pip ipython matplotlib pandas

Then and every time before using SWIMpy, activate the environment like this:

.. code-block:: console

    $ source activate swimpyenv


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

    $ pip install -e swimpy


`swimpy` in the above is the downloaded directory/repository.

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
