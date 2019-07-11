import pandas as pd


class Stations:
    def test_dataframe(self):
        self.assertTrue(hasattr(self.project, 'stations'))
        stations = self.project.stations
        self.assertIn(pd.DataFrame, stations.__class__.__mro__)

    def test_daily_discharge_observed(self):
        stations = self.project.stations
        self.assertTrue(hasattr(stations, 'daily_discharge_observed'))
        daily_discharge_obs = stations.daily_discharge_observed
        self.assertTrue(isinstance(daily_discharge_obs.index, pd.PeriodIndex))
        self.assertEqual(daily_discharge_obs.index.freq.freqstr, 'D')
