SWIMpy
======


A python package to interact and test the ecohydrological model SWIM.


* Version: 0.5.0
* Free software: MIT license
* Documentation: http://www.pik-potsdam.de/~wortmann/swimpy


Quickstart
----------

1. Setup python environment ``$ virtualenv swimpyenv`` and activate it
   ``$ source swimpyenv/bin/activate``
2. Download and install SWIMpy:
   ``$ pip install git+https://gitlab.pik-potsdam.de/wortmann/swimpy.git``
3. Setup your project in your model directory: ``$ swimpy setup``
4. Check the commandline help ``swimpy -h``, use your project in python
   scripts by importing ``swimpy`` and loading the project instance:
   ``project = swimpy.Project()`` or by starting the browser app
   ``$ swimpy browser start`` and navigate your browser to [http://localhost:8000](http://localhost:8000)


Features
--------

* Python API
* Documented commandline interface
* A central project settings file
* Easy per-project customisations and extensions
* Record runs with parameter changes, result indicators and data files
* Simple browser app to browse saved model runs and show plots
* Easy parameter reading/setting and output file reading
* Result visualisation with reusable matplotlib plot functions
* Interface to GRASS database
* Linking to the 
  [evoalgos](https://ls11-www.cs.tu-dortmund.de/people/swessing/evoalgos/doc/index.html)
  multiobjective evolutionary optimization package
