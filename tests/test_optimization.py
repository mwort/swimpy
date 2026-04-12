"""Tests for evolutionary optimisation algorithms (slow, requires swim run)."""
from __future__ import print_function, absolute_import
import os.path as osp

import pytest

from conftest import ProjectTestCase


OBJECTIVES = ['station_daily_discharge.rNSE.BLANKENSTEIN',
              'station_daily_discharge.pbias_abs.BLANKENSTEIN']
PARAMETERS = {'smrate0': (0.2, 0.7),
              'sccor0': (0.1, 10),
              'ecal0': (0.7, 1.3),
              'roc2_0': (0.5, 10)}


@pytest.mark.slow
class TestEvoalgos(ProjectTestCase):

    outputfile = 's01_SMSEMOA_populations.csv'
    algorithm_kwargs = {
        "parameters": PARAMETERS,
        "objectives": OBJECTIVES,
        "population_size": 4,
        "max_generations": 3,
        "prefix": 's01'
    }
    populations = None
    plot_functions = ['plot_generation_objectives', 'plot_objective_scatter',
                      'plot_parameter_distribution']

    @classmethod
    def setup_class(cls):
        # run with multiprocessing to also work on a single machine
        cls.project.settings(cluster_run_parallel_parallelism='mp')
        cls.output = osp.join(cls.project.projectdir, cls.outputfile)
        # only run algorithm if output does not already exist (speed up reruns)
        if not osp.exists(cls.output):
            cls.project.config_parameters(nbyr=2)
            cls.project.SMSEMOA(**cls.algorithm_kwargs)
        cls.populations = cls.project.SMSEMOA.read_populations(cls.output)

    def test_output(self):
        """Only makes sense if algorithm was run."""
        assert len(self.project.clone.names()) == 0
        assert osp.exists(self.output)
        output_pops = self.project.SMSEMOA.read_populations(self.output)
        run_pops = self.populations
        assert list(output_pops.columns) == list(run_pops.columns)
        for pops in [output_pops, run_pops]:
            assert len(pops) == 16
            assert pops.objectives == sorted(OBJECTIVES)
            assert pops.parameters == sorted(PARAMETERS.keys())

    def test_plots(self):
        pops = self.populations
        args = {'plot_objective_scatter': dict(best=True)}
        for pf in self.plot_functions:
            opath = osp.join(self.project.projectdir, pf + '.png')
            getattr(pops, pf)(output=opath, **(args.get(pf, {})))
            assert osp.exists(opath)
