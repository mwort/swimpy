"""Tests for swimpy package."""

import os.path as osp

import pandas as pd


class Parameters:

    def test_basin_parameters(self):
        bsn = self.project.basin_parameters()
        self.assertEqual(type(bsn), dict)
        self.assertGreater(len(bsn), 0)
        for k, v in bsn.items():
            self.assertEqual(self.project.basin_parameters(k), v)

    def test_config_parameters(self):
        cod = self.project.config_parameters()
        self.assertEqual(type(cod), dict)
        self.assertGreater(len(cod), 0)
        for k, v in cod.items():
            self.assertEqual(self.project.config_parameters(k), v)

    def test_subcatch_parameters(self):
        # read
        sbc = self.project.subcatch_parameters()
        self.assertIsInstance(sbc, pd.DataFrame)
        BLKS = self.project.subcatch_parameters('BLANKENSTEIN')
        self.assertIsInstance(BLKS, pd.Series)
        roc2 = self.project.subcatch_parameters('roc2')
        self.assertIsInstance(roc2, pd.Series)
        # write
        self.project.subcatch_parameters(roc2=1)
        self.assertEqual(self.project.subcatch_parameters('roc2').mean(), 1)
        self.project.subcatch_parameters(BLANKENSTEIN=2)
        BLKS = self.project.subcatch_parameters('BLANKENSTEIN').mean()
        self.assertEqual(BLKS, 2)
        newparamdict = {'roc2': 3.0, 'roc4': 10.0}
        HOF = self.project.subcatch_parameters('HOF')
        for k, v in newparamdict.items():
            HOF[k] = v
        self.project.subcatch_parameters(HOF=newparamdict)
        self.assertTrue((self.project.subcatch_parameters('HOF') == HOF).all())
        # write entire DataFrame
        self.project.subcatch_parameters(sbc.copy())
        nsbc = self.project.subcatch_parameters()
        self.assertTrue((nsbc == sbc).all().all())

    def test_changed_parameters(self):
        verbose = False
        from random import random
        original = self.project.changed_parameters(verbose=verbose)
        bsn = self.project.basin_parameters()
        scp = self.project.subcatch_parameters().T.stack().to_dict()
        nametags = [(k, None) for k in bsn] + scp.keys()
        nametags_original = [(e['name'], e['tags']) for e in original]
        for nt in nametags:
            self.assertIn(nt, nametags_original)
        run = self.project.browser.insert('run')
        for attr in original:
            self.project.browser.insert('parameter', run=run, **attr)
        self.project.basin_parameters(roc4=random(), da=random()*1000)
        changed = self.project.changed_parameters(verbose=verbose)
        self.assertEqual(sorted([e['name'] for e in changed]), ['da', 'roc4'])
        self.project.basin_parameters(**bsn)
        self.assertEqual(self.project.changed_parameters(verbose=verbose), [])
        self.project.subcatch_parameters(roc4=random())
        changed = self.project.changed_parameters(verbose=verbose)
        expresult = [('roc4', 'BLANKENSTEIN'), ('roc4', 'HOF')]
        nametags = sorted([(e['name'], e['tags']) for e in changed])
        self.assertEqual(nametags, expresult)


class Processing:
    def test_cluster_run(self):
        self.project.submit_cluster('testjob', 'run', dryrun=True, somearg=123)
        jfp = osp.join(self.project.resourcedir, 'cluster', 'testjob.py')
        self.assertTrue(osp.exists(jfp))

    def test_save_run(self):
        # test indicators and files
        indicators = ['indicator1', 'indicator2']
        ri_functions = [lambda p: 5,
                        lambda p: {'HOF': 0.1, 'BLANKENSTEIN': 0.2}]
        ri_values = {i: f(None) for i, f in zip(indicators, ri_functions)}
        files = ['file1', 'file2']
        somefile = osp.join(osp.dirname(__file__), '__init__.py')
        rf_functions = [lambda p: pd.DataFrame(range(100)),
                        lambda p: {'HOF': file(__file__),
                                   'BLANKENSTEIN': somefile}]
        rf_values = {i: f(None) for i, f in zip(files, rf_functions)}

        def check_files(fileobjects):
            self.assertEqual(len(fileobjects), 3)
            fdir = osp.join(self.project.browser.settings.filesdir, 'runs')
            for fo in fileobjects:
                self.assertTrue(osp.exists(fo.file.path))
                self.assertTrue(fo.file.path.startswith(fdir))
                self.assertIn(fo.tags.split()[0], files)
            return

        def check_indicators(indicatorobjects):
            self.assertEqual(len(indicatorobjects), 3)
            for io in indicatorobjects:
                self.assertIn(io.name, indicators)
            return
        # save run without any files or indicators
        run = self.project.save_run(notes='Some run notes',
                                    tags='testing test')
        self.assertIsInstance(run, self.project.browser.models['run'])
        self.assertTrue(hasattr(run, 'notes'))
        self.assertIn('test', run.tags.split())
        # pass indicators + files to save_run
        run = self.project.save_run(indicators=ri_values, files=rf_values)
        check_indicators(run.resultindicators.all())
        check_files(run.resultfiles.all())
        # pass as settings variables
        self.project.settings(**dict(zip(indicators, ri_functions)))
        self.project.settings(**dict(zip(files, rf_functions)))
        self.project.settings(resultfile_functions=files,
                              resultindicator_functions=indicators)
        run = self.project.save_run()
        check_indicators(run.resultindicators.all())
        check_files(run.resultfiles.all())


class Run:
    def test_project_run_data(self):
        from swimpy import results
        for r in results.properties.keys():
            df_project = getattr(self.project, r)
            self.project.settings(resultfile_functions=[r])
            run = self.project.save_run(notes='test saved ' + r)
            df_run = getattr(run, r)
            self.assertTrue(all(df_project.index == df_run.index))
            self.assertTrue(all(df_project.columns == df_run.columns))
            self.assertAlmostEqual((df_project-df_run).sum().sum(), 0)
