import os.path as osp
import warnings
from shutil import which


class Cluster:

    def test_cluster_run(self):
        self.project.cluster('testjob', 'run', dryrun=True, somearg=123)
        jfp = osp.join(self.project.cluster.resourcedir, 'testjob.py')
        self.assertTrue(osp.exists(jfp))
        self.project.run(cluster=dict(jobname='runtestjob', dryrun=True))
        jfp = osp.join(self.project.cluster.resourcedir, 'runtestjob.py')
        self.assertTrue(osp.exists(jfp))

    def run_parallel(self, parallelism):
        oyrs = self.project.config_parameters['nbyr']
        self.project.config_parameters(nbyr=2)
        args = [dict(smrate=i) for i in [0.1, 0.3, 0.6]]
        runs = self.project.cluster.run_parallel(
                clones=2, args=args, prefix='test', parallelism=parallelism,
                time=1)
        self.assertEqual(runs.count(), 3)
        clones = [self.project.clone[c] for c in self.project.clone.names()
                  if c.startswith('test')]
        self.assertEqual(len(clones), 2)
        # just run clones again
        runs2 = self.project.cluster.run_parallel(
                    clones, prefix='test2', parallelism=parallelism, time=1)
        self.assertEqual(runs2.count(), 2)
        self.project.config_parameters(nbyr=oyrs)

    def test_run_parallel_jobs(self):
        if which('sbatch'):
            self.run_parallel('jobs')
        else:
            warnings.warn(
                'Cant test cluster.run_parallel with jobs without sbatch.')

    def test_run_parallel_mp(self):
        self.run_parallel('mp')
