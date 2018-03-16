"""
Default SWIMpy project settings.

They can be overriden in the settings.py file or when creating a project using
Project(setting=value)
"""
from modelmanager.plugins import Browser, Clone, Templates
from .output import *
from .input import *

# input files
subcatch_parameter_file = 'input/subcatch.prm'

swim = './swim'

# cluster SLURM arguments
slurmargs = {'qos': 'short', 'account': 'swim'}
