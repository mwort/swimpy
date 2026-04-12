"""Tests for swimpy model run and cluster processing."""
import os
import os.path as osp

import pandas as pd
import pytest

from conftest import ProjectTestCase
from swimpy.tests import test_running


class TestProcessing(ProjectTestCase, test_running.Cluster):

    def test_save_run(self):
        indicators = ['indicator1', 'indicator2']
        ri_functions = [lambda p: 5,
                        lambda p: {'HOF': 0.1, 'BLANKENSTEIN': 0.2}]
        ri_values = {i: f(None) for i, f in zip(indicators, ri_functions)}

        files = ['file1', 'file2']
        somefile = osp.join(osp.dirname(__file__), 'test_processing.py')
        rf_functions = [lambda p: pd.DataFrame(list(range(100))),
                        lambda p: {'HOF': open(__file__),
                                   'BLANKENSTEIN': somefile}]
        rf_values = {i: f(None) for i, f in zip(files, rf_functions)}

        def check_files(fileobjects):
            assert len(fileobjects) == 3
            fdir = osp.join(self.project.browser.settings.filesdir, 'runs')
            for fo in fileobjects:
                assert osp.exists(fo.file.path)
                assert fo.file.path.startswith(fdir)
                assert fo.tags.split()[0] in files

        def check_indicators(indicatorobjects):
            assert len(indicatorobjects) == 3
            for io in indicatorobjects:
                assert io.name in indicators

        run = self.project.save_run(notes='Some run notes', tags='testing test')
        assert isinstance(run, self.project.browser.models['run'])
        assert hasattr(run, 'notes')
        assert 'test' in run.tags.split()

        run = self.project.save_run(indicators=ri_values, files=rf_values)
        check_indicators(run.indicators.all())
        check_files(run.files.all())

        self.project.settings(**dict(zip(indicators, ri_functions)))
        self.project.settings(**dict(zip(files, rf_functions)))
        self.project.settings(save_run_files=files,
                              save_run_indicators=indicators)
        run = self.project.save_run()
        check_indicators(run.indicators.all())
        check_files(run.files.all())

    @pytest.mark.slow
    def test_run(self):
        logfile = osp.join(self.project.outputpath, 'swim.log')
        if osp.exists(logfile):
            os.remove(logfile)
        self.project.run(save=False, quiet=True)
        assert osp.exists(logfile)
