PY = python

all: clean default test_hydro test_grass test_optimization

default:
	$(PY) tests.py -v

clean:
	git clean -f -x ./*

test_%:
	$(PY) $@.py

coverage:
	$(MAKE) PY='coverage run -a' all
	coverage html -i --include=../swimpy/*
	rm .coverage
