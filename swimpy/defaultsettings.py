"""
Default project settings.

They can be overriden in the `settings.py` file or temporarily when
instantiating a project using `p = Project(setting=value)`.
"""
# local imports
from modelmanager.utils import propertyplugin as _propertyplugin

# plugins
from modelmanager.plugins import Browser, Clone, Templates
from swimpy.grass import Subbasins, Hydrotopes, Routing, Substats
from swimpy.input import *
from swimpy.output import *
from swimpy.tests import Tests

# SWIM executable
swim = './swim'

# Cluster SLURM arguments
slurmargs = {'qos': 'short', 'account': 'swim'}
