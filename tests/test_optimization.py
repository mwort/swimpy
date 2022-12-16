from __future__ import print_function, absolute_import
import os.path as osp
import unittest
import cProfile, pstats

from tests import ProjectTestCase


OBJECTIVES = ['subbasin_label_daily_selected_stations_discharge.rNSE.BLANKENSTEIN',
              'subbasin_label_daily_selected_stations_discharge.pbias_abs.BLANKENSTEIN']
PARAMETERS = {'smrate': (0.2, 0.7),
              'sccor': (0.1, 10),
              'ecal': (0.7, 1.3),
              'roc2': (0.5, 10)}


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
    def setUpClass(self):
        super(TestEvoalgos, self).setUpClass()
        # run with multiprocessing to also run on single machine
        self.project.settings(cluster_run_parallel_parallelism='mp')
        self.output = osp.join(self.project.projectdir, self.outputfile)
        # only run algorithm if output doesnt exist to speed up output tests
        if not osp.exists(self.output):
            self.project.config_parameters(nbyr=2)
            run = self.project.SMSEMOA(**self.algorithm_kwargs)
            # TODO: check why next line does not work; in meantime use workaround
            # self.populations = run.optimization_populations
            self.populations = self.project.SMSEMOA.read_populations(
                                self.output)
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
        args = {'plot_objective_scatter': dict(best=True)}
        for pf in self.plot_functions:
            opath = osp.join(self.project.projectdir, pf+'.png')
            getattr(pops, pf)(output=opath, **(args[pf] if pf in args else {}))
            self.assertTrue(osp.exists(opath))


if __name__ == '__main__':
    cProfile.run('unittest.main()', 'pstats')
    # print profile stats ordered by time
    pstats.Stats('pstats').strip_dirs().sort_stats('time').print_stats(5)
