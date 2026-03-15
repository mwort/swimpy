Automatic, multi-objective calibration on the PIK cluster
=========================================================

This example shows how to run ready setup autocalibration in the SWIM Blankenstein
test project in the hope that it will be useful to transfer for other projects.
This example works entirely with preinstalled software (incl. SWIMpy) on the
PIK cluster and uses as little Python code as possible, i.e. mainly bash commands.

Prerequisite
------------
The autocalibration setup is committed to a SWIM branch and can be cloned/downloaded
like this (will prompt for your PIK gitlab username and password).

.. code-block:: console

    $ module load git/2.16.1
    $ git clone https://gitlab.pik-potsdam.de/wortmann/swim.git --branch swimpy_autocalibration
    $ cd swim

Once downloaded, compile SWIM as usual and doublecheck if the Blankenstein project
is running as expected:

.. code-block:: console

    $ module load compiler/gnu/7.3.0
    $ cd code
    $ make
    $ cd ../project
    $ make

We need SWIMpy (and all its dependencies) to run the calibration, i.e. either follow
:doc:`/installation` or activate the preinstalled conda environment like this (your prompt
should be prefixed with ``(swimpyenv)`` from now on) and test it's working by showing
the help:

.. code-block:: console

    $ module load anaconda/5.0.0_py3
    $ source activate /home/wortmann/src/swimpy/swimpyenv
    $ swimpy -h


Settings
--------

All SWIMpy settings are given in the ``swimpy/settings.py`` in the Blankenstein ``project/``
and, for the purpose of this example, all autocalibration settings have been prepared there.
Give this a good read (``vim swimpy/settings.py``).

.. literalinclude:: autocalibration_settings.py

In this example we'll be using the `SMS-EMOA optimisation algorithm`_, a multiobjective,
evolutionary algorithm based on the maximisation of the dominated hypervolume of the
objective space (but there are others, such as the commonly used NSGA algorithm, see the
`evoalgos documentation`_ for details). All the settings prefixed with `SMSEMOA_` refer
to the algorithm settings and can also be parsed on the commandline (see
``swimpy SMSEMOA -h`` for details). The population_size and max_generations settings here
represent test values, commonly used values are 50-100 for both, totaling around 5000-10000
model runs per calibration. But this is not to say that even ~100 runs will already lead
to strong performance improvements. 

.. _`SMS-EMOA optimisation algorithm`: https://www.sciencedirect.com/science/article/abs/pii/S0377221706005443
.. _`evoalgos documentation`: https://www.simonwessing.de/evoalgos/doc/algo.html


Running the algorithm
---------------------

Since all required settings are given in the ``swimpy/settings.py`` file, running it only requires
(``--output`` has the default here):

.. code-block:: console

    $ swimpy SMSEMOA --output SMSEMOA_populations.csv

SWIMpy will now perform a test run of the model, the model parameter assignment and the retrieval
of the objective functions. If this test passes, it will start with the parallel model runs
according to the chosen parallelisation and writes out the population median and min. objectives to
the console and for all model runs to the output file.

Multi-process parallelisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If ``cluster_run_parallel_parallelism = "mp"``, all available CPUs of your machine are used to
run SWIM in parallel. Note that on the cluster login nodes, this defaults to 48, which will
overload the nodes. It is recommended to not use more than 16 multi-processes, as is e.g. the
case on a single compute node.

SLURM job parallelisation
^^^^^^^^^^^^^^^^^^^^^^^^^
If ``cluster_run_parallel_parallelism = "jobs"``, each model run is submitted as SLURM job. There
is virtually now limit on the number of parallel runs, however, the jobs will have to pass
through the SLURM queue, which can get busy and as you accumulate CPU hours, your priority will
shrink, thus increasing your queuing time.

MPI parallelisation
^^^^^^^^^^^^^^^^^^^
If ``cluster_run_parallel_parallelism = "mpi"``, model runs are performed on distributed CPUs
on the cluster. This method, however, requires more setup and a specific submission command.
The advantage is that you are neither bound by node CPUs nor by the SLURM queue.
(... will document this in more detail later ...)


Visualising results
-------------------
(... to be continued, see :docs:`examples/mulitobjective_optimisation` for a Python-based
visualisation ...) 