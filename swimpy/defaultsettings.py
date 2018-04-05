"""
Default project settings.

They can be overriden in the `settings.py` file or temporarily when
instantiating a project using `p = Project(setting=value)`.
"""
# local imports
from modelmanager.utils import propertyplugin as _propertyplugin
from swimpy import input, output

# plugins
from modelmanager.plugins import Browser, Clone, Templates
from swimpy.grass import Subbasins, Hydrotopes, Routing, Substats
from swimpy.tests import Tests
from swimpy.output import output

# SWIM executable
swim = './swim'

# Cluster SLURM arguments
slurmargs = {'qos': 'short', 'account': 'swim'}

# input properties
basin_parameters = _propertyplugin(input.basin_parameters)
config_parameters = _propertyplugin(input.config_parameters)
subcatch_parameters = _propertyplugin(input.subcatch_parameters)
