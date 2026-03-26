PY = python
DEPENDENCIES = ../dependencies/swim/project/output


default: $(DEPENDENCIES)
	$(PY) tests.py -v

all: clean default hydro grass optimization

clean:
	git clean -f -x ./*

mpi:
	srun -n 16 --mpi=pmi2 --qos=priority python test_mpi.py

setup: $(DEPENDENCIES)
	$(PY) tests.py setup

coverage:
	$(MAKE) PY='coverage run -a' all
	coverage html -i --include=../swimpy/*
	rm .coverage

%: $(DEPENDENCIES)
	$(PY) test_$@.py
