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


class Processing:
    def test_cluster_run(self):
        self.project.submit_cluster('testjob', 'run', dryrun=True, somearg=123)
        jfp = osp.join(self.project.resourcedir, 'cluster', 'testjob.py')
        self.assertTrue(osp.exists(jfp))


class Run:
    def test_project_run_data(self):
        from swimpy.utils import ProjectOrRunData
        projectprops = self.project.settings.properties
        resultproperties = [n for n, p in projectprops.items()
                            if (hasattr(p, 'isplugin') and
                                ProjectOrRunData in p.plugin_class.__bases__)]
        self.assertGreater(len(resultproperties), 0)
        for r in sorted(resultproperties):
            df_project = getattr(self.project, r)
            self.project.settings(resultfile_functions=[r])
            run = self.project.save_run(notes='test saved ' + r)
            df_run = getattr(run, r)
            self.assertTrue(all(df_project.index == df_run.index))
            self.assertTrue(all(df_project.columns == df_run.columns))
            self.assertAlmostEqual((df_project-df_run).sum().sum(), 0)
