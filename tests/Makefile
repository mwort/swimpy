PY = python

all: default

default:
	$(PY) tests.py -v

clean:
	git clean -f -x ./*

coverage:
	$(MAKE) PY='coverage run -a' all
	coverage html -i --include=../modelmanager/*
	rm .coverage
