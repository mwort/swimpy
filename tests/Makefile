PY = python


default:
	$(PY) tests.py -v

all: clean default hydro grass optimization

clean:
	git clean -f -x ./*

mpi:
	srun -n 16 --mpi=pmi2 --qos=priority python test_mpi.py

setup: #clean
	$(PY) tests.py setup
	$(eval $@_dirs := $(shell cd ../dashboard_resources/project; find . -type d))
	$(eval $@_files := $(shell cd ../dashboard_resources/project; find . -type f) \
		$(shell cd ../dashboard_resources/project; find . -type l))
	cd project; mkdir -p $($@_dirs)
	@cd project; for f in $($@_files) ; do \
		rm -f $$f ; \
		ln -s $$(realpath --relative-to=$$(dirname $$f) ../../dashboard_resources/project/$$f) $$f ; \
	done
	cd project; swimpy config_parameters --set climatedir=./climate/
	cd project; swimpy basin_parameters --set bRunoffDat=0



coverage:
	$(MAKE) PY='coverage run -a' all
	coverage html -i --include=../swimpy/*
	rm .coverage

%:
	$(PY) test_$@.py
