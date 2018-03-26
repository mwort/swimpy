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

# SWIM executable
swim = './swim'

# Cluster SLURM arguments
slurmargs = {'qos': 'short', 'account': 'swim'}

# input properties
basin_parameters = _propertyplugin(input.basin_parameters)
config_parameters = _propertyplugin(input.config_parameters)
subcatch_parameters = _propertyplugin(input.subcatch_parameters)

# output properties
station_daily_discharge = _propertyplugin(output.station_daily_discharge)
subbasin_daily_waterbalance = _propertyplugin(output.subbasin_daily_waterbalance)
catchment_daily_waterbalance = _propertyplugin(output.catchment_daily_waterbalance)
catchment_monthly_waterbalance = _propertyplugin(output.catchment_monthly_waterbalance)
catchment_annual_waterbalance = _propertyplugin(output.catchment_annual_waterbalance)
