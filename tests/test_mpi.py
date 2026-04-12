"""Tests for MPI parallel runs (requires mpi4py and srun).

Run via: srun -n 16 --mpi=pmi2 --qos=priority pytest tests/test_mpi.py -m mpi
See https://gitlab.pik-potsdam.de/linstead/test-project/-/blob/8de47fff/SLURM.md#mpi-and-python-mpi4py
"""
import os.path as osp
import shutil

import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI

import swimpy
from conftest import (ProjectTestCase, SWIM_TEST_PROJECT, SWIM_REPO_PROJECT,
                      MSWIM_GRASSDB, TEST_GRASSDB, TEST_SETTINGS,
                      _rmtree_nfs)

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()


@pytest.mark.mpi
class TestMpi(ProjectTestCase):
    """MPI tests manage their own project setup to coordinate across ranks."""

    @classmethod
    def setup_class(cls):
        if RANK == 0:
            _rmtree_nfs(SWIM_TEST_PROJECT)
            _rmtree_nfs(TEST_GRASSDB)
            shutil.copytree(SWIM_REPO_PROJECT, SWIM_TEST_PROJECT,
                            dirs_exist_ok=True)
            if osp.exists(MSWIM_GRASSDB):
                shutil.copytree(MSWIM_GRASSDB, TEST_GRASSDB,
                                dirs_exist_ok=True)
            p = swimpy.project.setup(SWIM_TEST_PROJECT)
            shutil.copy(TEST_SETTINGS, p.settings.file)
            project = swimpy.Project(SWIM_TEST_PROJECT)
            project.browser.project.settings.load()
            project.config_parameters(nbyr=2)
            project.browser.settings.unset()
        COMM.Barrier()
        cls.project = swimpy.Project(SWIM_TEST_PROJECT)

    @classmethod
    def teardown_class(cls):
        cls.project.browser.settings.unset()
        COMM.Barrier()
        if RANK == 0:
            _rmtree_nfs(SWIM_TEST_PROJECT)
            _rmtree_nfs(TEST_GRASSDB)

    def test_run_parallel(self):
        n = int(SIZE * 1.5)
        args = [dict(smrate=i / 10.) for i in range(n)]
        runs = self.project.cluster.run_parallel(args=args, parallelism='mpi')
        assert runs.count() == n
        cloneids = set([
            int(t.split()[1].split('_')[-1])
            for t in sorted(runs.values_list('tags', flat=True))
        ])
        assert list(cloneids) == list(range(min(SIZE, n)))
