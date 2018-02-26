from modelmanager.plugins import Browser, Clones, Templates

# input files
subcatch_parameter_file = 'input/subcatch.prm'

swim = './swim'

# cluster SLURM arguments
slurmargs = {'qos': 'short', 'account': 'swim'}
