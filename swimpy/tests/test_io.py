import os
import os.path as osp

import pandas as pd


class Parameters:

    def test_basin_parameters(self):
        bsn = self.project.basin_parameters
        self.assertGreater(len(bsn), 0)
        for k, v in bsn.items():
            self.assertEqual(self.project.basin_parameters(k)[0], v)

    def test_config_parameters(self):
        cod = self.project.config_parameters
        self.assertGreater(len(cod), 0)
        for k in cod:
            self.assertEqual(self.project.config_parameters(k)[0],
                             self.project.config_parameters[k])


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


class Output:
    def test_read_save_output(self):
        """Read, save and retrieve all output interfaces."""
        resultproperties = [
            rf for rf in self.project.output_interfaces
            if self.project.settings.properties[rf].plugin.path]
        self.assertGreater(len(resultproperties), 0)
        run = self.project.save_run(files=resultproperties)
        for r in sorted(resultproperties):
            print(r)
            df_project = getattr(self.project, r)
            df_run = getattr(run, r)
            self.assertTrue(all(df_project.index == df_run.index), r)
            self.assertTrue(all(df_project.columns == df_run.columns), r)
            sud = (df_project.copy()-df_run.copy()).sum().sum()
            self.assertAlmostEqual(sud, 0, msg=r)


class output_sums:
    """Compare the column sums of two SWIM runs.

    First run ``swimpy test output_sums -t create`` after the benchmark run.
    Then rerun SWIM with changes and run
    ``swimpy test output_sums -t compare``.
    """
    resourcedir_name = 'test_output_sums'
    precision = 5

    def test_create(self):
        """Create benchmark files (required before compare)."""
        for o in self.project.output_interfaces:
            df = getattr(self.project, o)
            if df.path:
                path = self._benchmark_path(df.path)
                os.makedirs(osp.dirname(path), exist_ok=True)
                print('Writing %s' % path)
                df.sum().to_pickle(path)

    def test_compare(self):
        """Compare output sums against benchmark."""
        for i in self.project.output_interfaces:
            df = getattr(self.project, i)
            if df.path is None:
                continue
            bpth = self._benchmark_path(df.path)
            with self.subTest(benchmark_path=bpth):
                self.assertTrue(osp.exists(bpth))
                benchmark = pd.read_pickle(bpth)
                # check each column sum
                for n, c in df.sum().items():
                    with self.subTest(path=df.path, column=n):
                        self.assertIn(n, benchmark)
                        b = benchmark[n]
                        # deviation for reporting
                        di = '%s%%' % (((c/b)-1)*100) if b else b-c
                        with self.subTest(
                                path=df.path, column=n, deviation=di):
                            self.assertAlmostEqual(
                                c, benchmark[n], self.precision)

    @property
    def resourcedir(self):
        return osp.join(self.project.resourcedir, self.resourcedir_name)

    def _benchmark_path(self, interface_path):
        odir = osp.join(self.project.projectdir, 'output')
        opath = osp.relpath(interface_path, odir)
        path = osp.join(self.resourcedir, opath)+'.pd'
        return path
