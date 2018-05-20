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

# defaults when saving a figure used in plot.plot_function
save_figure_defaults = dict(
    bbox_inches='tight',
    pad_inches=0.03,
    dpi=200,
    size=(180, 120),  # mm
)
