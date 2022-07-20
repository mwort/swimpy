"""
Default project settings.

They can be overriden in the `settings.py` file or temporarily when
instantiating a project using `p = Project(setting=value)`.
"""

# import inspect
# from modelmanager.utils import propertyplugin as _propertyplugin
# from modelmanager.plugins.templates import TemplatesDict as _TemplatesDict
# from modelmanager.plugins.pandas import ReadWriteDataFrame as _ReadWriteDataFrame

# plugins
from modelmanager.plugins.browser import browser
from modelmanager.plugins import clone
from modelmanager.plugins import templates
from swimpy.grass import subbasins, hydrotopes, routing, substats
from swimpy import input
from swimpy import output
from swimpy.tests import test
from swimpy.utils import StationsUnconfigured as stations
from swimpy.utils import cluster
from swimpy.plot import plot_summary
from swimpy.optimization import SMSEMOA, CommaEA, NSGA2b, CMSAES

#: SWIM executable
swim = './swim'

#: Cluster SLURM arguments
cluster_slurmargs = {'qos': 'short', 'account': 'swim'}

#: Defaults when saving a figure used in plot.plot_function
save_figure_defaults = dict(
    bbox_inches='tight',
    pad_inches=0.03,
    dpi=200,
    size=(180, 120),  # mm
)

#: Default plots to show on :meth:`swimpy.project.Project.plot_summary` and
#: :meth:`browser.Run.plot_summary`
plot_summary_functions = ['station_daily_discharge.plot',
                          [('station_daily_discharge.plot', {'freq': 'a'}),
                           'station_daily_discharge.plot_regime'],
                          'catchment_annual_waterbalance.plot_mean',
                          ]


#: All input/output file interface plugins as propertyplugins
globals().update(input.PLUGINS)
globals().update(output.PLUGINS)

# for n, p in input.__dict__.items():
#     if inspect.isclass(p) and set([_ReadWriteDataFrame, _TemplatesDict]) & set(p.__mro__[1:]):
#         globals().update({n: _propertyplugin(p)})
# globals().update({n: input.__dict__[n] for n in ['climate']})


# put here to enable overriding
@property
def project_name(self):
    """Short SWIM project name inferred from *.nml file."""
    return self.parfile.replace('.nml', '')


#: Plugins that require a resource dir to exist.
#: Will not be loaded if resourcedir=False.
plugins_require_resourcedir = [
    'templates',
    'stations',
    'cluster',
    'clone',
    'browser'
]
