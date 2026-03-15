"""swimpy settings file.

This file accepts standard python code. See the docs for further details
(http://www.pik-potsdam.de/~wortmann/swimpy/usage.html#project-settings).
"""


#: Parameters from the .bsn file, (lower, upper) bounds
SMSEMOA_parameters = {
    'bff': (0, 1),
    'sccor': (0.1, 10),
    'roc2': (1, 20),
    'roc4': (1, 20),
    'smrate': (0., 1),
    'ecal': (0.8, 1.2),
    'xgrad1': (0., 0.001),
}

#: Objective functions renamed to short versions
#: This could be any function or attribute of the swimpy project, eg.
#: a function defined in this file.
SMSEMOA_objectives = {
    'rNSE': 'station_daily_discharge.rNSE.BLANKENSTEIN',
    'bias': 'station_daily_discharge.pbias_abs.BLANKENSTEIN',
}

#: Population (parallel model runs) and generations to run.
SMSEMOA_population_size = 16
SMSEMOA_max_generations = 10

#: try to restart from existing output file
SMSEMOA_restart = True

#: How to perform parallel model runs
#: mp : multiprocessing on the available CPUs of machine
#: jobs : send each model run as a SLURM job
#: mpi : MPI parallelisation
cluster_run_parallel_parallelism = "mp"

#: If you are running with jobs and you want to control any sbatch parameters
#: (e.g. qos, account, cpus-per-task), give them here. Note that the time
#: parameter is overridden and adapted to the model runtime.
cluster_slurmargs = {'account': 'swim'}
