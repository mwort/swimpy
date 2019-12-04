==========
Change log
==========

0.5 (2019-12-02)
----------------
* MPI support to `cluster.run_parallel` and optional `jobs` or `mp` parallelism
* Allow restarting optimization jobs
* Record `run_time` in runs table
* `basin_parameters.set_defaults` added
* input and output interface plugins can now be instatiated by path without project
* Generalisation of `hydro.gumbel_recurrence` to scipy distribution recurrence `dist_recurrence`
* `grass.GrassAttributeTable` exposed from `modelmanager.plugins.grass`


0.4 (2019-07-11)
----------------
* `tests` plugin renamed to `test`
* Added more project-independent tests (water balance)
* Lots of updating with SWIM/develop as m.swim/master
* Various improvements and fixes


0.3 (2019-03-19)
----------------
* Add optimization algorithms to default settings and include the `evoalgos`
  package to dependencies
* Allow default `stations.daily_discharge_observed` loading through standardised
  CSV file in resource directory
* Improved time management in optimization runs
* Add crop output file interfaces
* Allow project-specific tests in any file named `swimpy/test_*.py`
* bug fixes and refactoring


0.2 (2019-02-25)
----------------
* Extended setup to copy input files from SWIM's test project
* Separate `swimpy.hydro` module to bundle hydrology-related functionality
* Various new output file interfaces
* Bug fixes


0.1 (2018-10-09)
-----------------
* First release with all features listed in README
