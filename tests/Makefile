PY = python


default:
	$(PY) tests.py -v

all: clean default hydro grass optimization

clean:
	git clean -f -x ./*

mpi:
	srun -n 16 --mpi=pmi2 --qos=priority python test_mpi.py

setup:
	$(PY) tests.py setup

coverage:
	$(MAKE) PY='coverage run -a' all
	coverage html -i --include=../swimpy/*
	rm .coverage

%:
	$(PY) test_$@.py
