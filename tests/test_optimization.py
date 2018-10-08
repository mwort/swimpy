from __future__ import print_function, absolute_import
import os
import os.path as osp
import sys
import subprocess
import shutil
import unittest
import cProfile, pstats

import pandas as pd

from tests import ProjectTestCase, TEST_GRASSDB
from swimpy.optimization import SMSEMOA


OBJECTIVES = ['station_daily_discharge.rNSE.BLANKENSTEIN',
              'station_daily_discharge.pbias_abs.BLANKENSTEIN']
PARAMETERS = {'smrate': (0.2, 0.7),
              'sccor': (0.1, 10),
              'ecal': (0.7, 1.3),
              'roc2': (0.5, 10)}


class TestEvoalgos(ProjectTestCase):

    output='s01_SMSEMOA_populations.csv'
    algorithm_kwargs = {
        "parameters": PARAMETERS,
        "objectives": OBJECTIVES,
        "population_size": 4,
        "max_generations": 3,
        "prefix": 's01'
    }
    populations = None

    @classmethod
    def setUpClass(self):
        super(TestEvoalgos, self).setUpClass()
        os.chdir(self.project.projectdir)
        self.project.settings(SMSEMOA)
        # add to browser project instance too
        from django.conf import settings
        settings.PROJECT.settings(SMSEMOA)
        # only run algorithm if output doesnt exist to speed up output tests
        if not osp.exists(self.output):
            self.project.config_parameters(nbyr=2)
            self.project.basin_parameters(subcatch=0)
            run = self.project.SMSEMOA(**self.algorithm_kwargs)
            self.populations = run.optimization_populations
        else:
            self.populations = self.project.SMSEMOA.read_populations(
                                self.output)

    def test_output(self):
        """Only makes sense if algorithm was run!"""
        self.assertEqual(len(self.project.clone.names()), 0)
        self.assertTrue(osp.exists(self.output))
        output_pops = self.project.SMSEMOA.read_populations(self.output)
        run_pops = self.populations
        self.assertEqual(list(output_pops.columns), list(run_pops.columns))
        for pops in [output_pops, run_pops]:
            self.assertEqual(len(pops), 16)
            self.assertEqual(pops.objectives, sorted(OBJECTIVES))
            self.assertEqual(pops.parameters, sorted(PARAMETERS.keys()))

    def test_plots(self):
        pops = self.populations
        functions = ['plot_generation_objectives', 'plot_objective_scatter',
                     'plot_parameter_distribution']
        args = {'plot_objective_scatter': dict(best=True)}
        for pf in functions:
            opath = pf+'.png'
            getattr(pops, pf)(output=opath, **(args[pf] if pf in args else {}))
            self.assertTrue(osp.exists(opath))


if __name__ == '__main__':
    cProfile.run('unittest.main()', 'pstats')
    # print profile stats ordered by time
    pstats.Stats('pstats').strip_dirs().sort_stats('time').print_stats(5)
