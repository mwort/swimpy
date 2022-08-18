import os
import os.path as osp
import shutil

import pandas as pd


class Parameters:

    def test_config_parameters(self):
        import f90nml
        cod = self.project.config_parameters
        self.assertGreater(len(cod), 0)
        for k in cod:
            self.assertEqual(self.project.config_parameters(k)[0],
                             self.project.config_parameters[k])
        self.assertEqual(self.project.config_parameters('iyr'),
                         self.project.time_parameters('iyr'))
        self.project.config_parameters['iyr'] += 1
        self.assertEqual(self.project.config_parameters('iyr'),
                         self.project.time_parameters('iyr'))
        self.project.time_parameters(iyr=1996)
        self.assertEqual(self.project.config_parameters('iyr'),
                         self.project.time_parameters('iyr'))
        nml = f90nml.read(osp.join(self.project.projectdir, self.project.parfile))
        self.assertEqual(self.project.time_parameters['iyr'],
                         nml['time_parameters']['iyr'])
        # test that set_default() works
        for gr in self.project.config_parameters.keys():
            for nl in self.project.config_parameters[gr].keys():
                if self.project.config_parameters[gr][nl] != self.project.config_parameters.defaults[gr][nl]:
                    self.project.config_parameters.set_default(nl)
                    self.assertEqual(self.project.config_parameters(nl)[0],
                                     self.project.config_parameters.defaults[gr][nl])
                    nml = f90nml.read(osp.join(self.project.projectdir, self.project.parfile))
                    self.assertEqual(self.project.config_parameters(nl)[0],
                                     nml[gr][nl])
                    break
        # behaviour for not implemented parameters
        self.assertRaises(KeyError, self.project.time_parameters, 'bla')
        self.assertRaises(KeyError, self.project.time_parameters, bla=5)
        with self.assertRaises(KeyError):
            self.project.config_parameters['bla']
        with self.assertRaises(KeyError):
            self.project.config_parameters['bla'] = 5
        self.assertRaises(KeyError, self.project.config_parameters,
                          bla_parameters={'bla1': 1995, 'bla2': True})
          

class Input:

    def test_discharge(self):
        ro = self.project.discharge
        if len(ro.subbasins):
            sbattr = self.project.subbasins.attributes
        for n in ro.columns:
            self.assertIn(n, self.project.stations.index)
            if len(ro.subbasins):
                self.assertIn(ro.subbasins[n], sbattr.index)


class Output:

    def test_output_files(self):
        outf = osp.join(self.project.output_files.path)
        outf_save = osp.join(self.project.inputpath, 'output_save.nml')
        self.project.output_files.write(outf_save)
        from swimpy.output import OutputFile as ofileclass
        # new output that does not yet exist
        self.assertFalse(hasattr(self.project, 'hydrotope_monthly_testevap'))
        self.project.output_files(hydrotope_monthly_testevap=['etp', 'eta'])
        self.assertTrue(hasattr(self.project, 'hydrotope_monthly_testevap'))
        self.assertTrue('hydrotope_monthly_testevap' in self.project.output_files.keys())
        testevap = self.project.hydrotope_monthly_testevap
        self.assertIsInstance(testevap, ofileclass)
        self.assertEqual(repr(testevap),
                         '<File output/hydrotope_monthly_testevap.csv does not yet exist. You need to run SWIM first.>')
        # implicit write() works
        self.project.output_files.read()
        self.assertTrue('hydrotope_monthly_testevap' in self.project.output_files.keys())
        # different output file associated
        self.project.output_files.read(outf_save)
        self.assertEqual(self.project.output_files.path, outf_save)
        self.assertFalse('hydrotope_monthly_testevap' in self.project.output_files.keys())
        shutil.move(outf_save, outf)

    # TODO: not needed anymore?
    # def test_read_save_output(self):
    #     """Read, save and retrieve all output interfaces."""
    #     resultproperties = [
    #         rf for rf in self.project.output_interfaces
    #         if self.project.settings.properties[rf].plugin.path]
    #     self.assertGreater(len(resultproperties), 0)
    #     run = self.project.save_run(files=resultproperties)
    #     for r in sorted(resultproperties):
    #         print(r)
    #         df_project = getattr(self.project, r)
    #         df_run = getattr(run, r)
    #         self.assertTrue(all(df_project.index == df_run.index), r)
    #         self.assertTrue(all(df_project.columns == df_run.columns), r)
    #         sud = (df_project.copy()-df_run.copy()).sum().sum()
    #         self.assertAlmostEqual(sud, 0, msg=r)


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
        for o in self.project.output_files:
            df = getattr(self.project, o)
            if df.path:
                path = self._benchmark_path(df.path)
                os.makedirs(osp.dirname(path), exist_ok=True)
                print('Writing %s' % path)
                df.sum().to_pickle(path)

    def test_compare(self):
        """Compare output sums against benchmark."""
        for i in self.project.output_files:
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
        opath = osp.relpath(interface_path, self.project.outputpath)
        path = osp.join(self.resourcedir, opath)+'.pd'
        return path
