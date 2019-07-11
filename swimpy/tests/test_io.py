
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
        resultproperties = [rf for rf in self.project.output_interfaces
                            if self.project.settings.properties[rf].plugin.path]
        self.assertGreater(len(resultproperties), 0)
        run = self.project.save_run(files=resultproperties)
        for r in sorted(resultproperties):
            print(r)
            df_project = getattr(self.project, r)
            df_run = getattr(run, r)
            self.assertTrue(all(df_project.index == df_run.index), r)
            self.assertTrue(all(df_project.columns == df_run.columns), r)
            self.assertAlmostEqual((df_project-df_run).sum().sum(), 0, msg=r)
