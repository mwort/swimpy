import unittest
import cProfile, pstats

import swimpy
import tests
from mpi4py import MPI


COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()


class TestMpi(tests.ProjectTestCase):
    @classmethod
    def setUpClass(self):
        if RANK == 0:
            super(TestMpi, self).setUpClass()
            self.project.config_parameters(nbyr=2)
        COMM.Barrier()
        self.project = swimpy.Project(tests.SWIM_TEST_PROJECT)

    @classmethod
    def tearDownClass(self):
        self.project.browser.settings.unset()
        COMM.Barrier()
        if RANK == 0:
            super(TestMpi, self).tearDownClass()

    def test_run_parallel(self):
        n = int(SIZE*1.5)
        args = [dict(smrate=i/10.) for i in range(n)]
        runs = self.project.cluster.run_parallel(
            args=args, parallelism='mpi')
        self.assertEqual(runs.count(), n)
        cloneids = set([int(t.split()[1].split('_')[-1])
                        for t in sorted(runs.values_list('tags', flat=True))])
        self.assertListEqual(list(cloneids), list(range(min(SIZE, n))))


if __name__ == '__main__':
    cProfile.run('unittest.main()', 'pstats')
    if RANK == 0:
        # print profile stats ordered by time
        pstats.Stats('pstats').strip_dirs().sort_stats('time').print_stats(5)
