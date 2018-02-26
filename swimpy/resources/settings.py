from modelmanager.plugins import Browser, Clone, Templates

# input files
subcatch_parameter_file = 'input/subcatch.prm'

swim = './swim'

# cluster SLURM arguments
slurmargs = {'qos': 'short', 'account': 'swim'}
