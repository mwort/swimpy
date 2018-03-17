"""
Default SWIMpy project settings.

They can be overriden in the settings.py file or when creating a project using
Project(setting=value)
"""
from modelmanager.plugins import Browser, Clone, Templates
from .output import *
from .input import *

swim = './swim'

# cluster SLURM arguments
slurmargs = {'qos': 'short', 'account': 'swim'}
