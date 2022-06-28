.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help


define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

ifeq ($(OS),Windows_NT)
	FILE_SEP=";"
else
	FILE_SEP=":"
endif

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

lint: ## check style with flake8
	flake8 swimpy tests

test: ## run tests quickly with the default Python
	cd tests; $(MAKE)

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source swimpy setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	#rm -f docs/swimpy.rst
	#rm -f docs/modules.rst
	#sphinx-apidoc -o docs/ swimpy
	$(MAKE) -C docs clean
	cd docs; python theproject.py
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## upload docs once versioned and unversioned
	rsync -crzP docs/_build/html/ wortmann@cluster.pik-potsdam.de:www/swimpy
	rsync -crzP docs/_build/html/ wortmann@cluster.pik-potsdam.de:www/swimpy/`python -c "import swimpy; print(swimpy.__version__)"`

release: clean ## package and upload a release
	python setup.py sdist upload
	python setup.py bdist_wheel upload

dist: clean dist/swimpy dist/swim dist/swimpy-dashboard ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

docker_build:
	make -C dependencies/m.swim/test clean
	make -C dependencies/swim/project clean
	make -C tests/ clean
	docker build -t mwort/swim:latest .

docker_push:
	docker push mwort/swim:latest

# Creates a single executable for a given platform, using swimpy/scripts/swimpy as the entrypoint
# Meant to be ran in a venv or docker container
dist/swimpy: ## build single executable for swimpy
	pip install pyinstaller
	pip install -e .
	pip install -r requirements_dev.txt
	pyinstaller \
		-p dependencies/modelmanager \
		-p dependencies/m.swim \
		-p . \
		-F \
		--collect-submodules dependencies \
		-d noarchive \
		--add-data dependencies/modelmanager/modelmanager/$(FILE_SEP)modelmanager \
		swimpy/scripts/swimpy

# Creates a single executable for a given platform, using swimpy/scripts/swimpy-dashboard as the entrypoint
# Meant to be created in a venv or docker container
dist/swimpy-dashboard: dependencies/swim/code/swim ## build single executable for `swimpy dashboard start`
	pip install pyinstaller
	pip install -e .[dashboard]
	pip install -r requirements_dev.txt
	pyinstaller \
		-p dependencies/modelmanager \
		-p dependencies/m.swim \
		-p . \
		-F \
		--collect-submodules dependencies \
		-d noarchive \
		--add-data dependencies/modelmanager/modelmanager/$(FILE_SEP)modelmanager \
		--add-data swimpy/$(FILE_SEP)swimpy/ \
		--add-data dependencies/swim/$(FILE_SEP)dependencie/swim/ \
		swimpy/scripts/swimpy-dashboard


dependencies/swim/code/swim:
	make -C dependencies/swim/code

dist/swim: dependencies/swim/code/swim
	mkdir -p dist/
	cp dependencies/swim/code/swim dist/swim
