"""Tests for swimpy package."""
import os.path as osp


class Parameters:

    def test_basin_parameters(self):
        bsn = self.project.basin_parameters
        self.assertGreater(len(bsn), 0)
        for k, v in bsn.items():
            self.assertEqual(self.project.basin_parameters(k)[0], v)

    def test_config_parameters(self):
        cod = self.project.config_parameters
        self.assertGreater(len(cod), 0)
        for k, v in cod.items():
            self.assertEqual(self.project.config_parameters(k)[0], v)


class Input:

    def test_structure_file(self):
        strf = self.project.hydrotopes.attributes
        akm = strf.area.sum()*1e-6
        self.assertAlmostEqual(akm, self.project.basin_parameters['da'], 1)
        cod_mb = self.project.config_parameters['mb']
        self.assertEqual(strf.subbasinID.max(), cod_mb)

    def test_station_daily_discharge_observed(self):
        ro = self.project.station_daily_discharge_observed
        if len(ro.subbasins):
            sbattr = self.project.subbasins.attributes
        for n in ro.columns:
            self.assertIn(n, self.project.stations.index)
            if len(ro.subbasins):
                self.assertIn(ro.subbasins[n], sbattr.index)


class Processing:
    def test_cluster_run(self):
        self.project.cluster('testjob', 'run', dryrun=True, somearg=123)
        jfp = osp.join(self.project.cluster.resourcedir, 'testjob.py')
        self.assertTrue(osp.exists(jfp))
        self.project.run(cluster=dict(jobname='runtestjob', dryrun=True))
        jfp = osp.join(self.project.cluster.resourcedir, 'runtestjob.py')
        self.assertTrue(osp.exists(jfp))


class Run:
    def test_project_run_data(self):
        resultproperties = self.project.resultfile_interfaces
        self.assertGreater(len(resultproperties), 0)
        for r in sorted(resultproperties):
            df_project = getattr(self.project, r)
            self.project.settings(save_run_files=[r])
            run = self.project.save_run(notes='test saved ' + r)
            df_run = getattr(run, r)
            self.assertTrue(all(df_project.index == df_run.index))
            self.assertTrue(all(df_project.columns == df_run.columns))
            self.assertAlmostEqual((df_project-df_run).sum().sum(), 0)
