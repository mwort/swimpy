DEPENDENCIES = ../dependencies/swim/project/output

default: $(DEPENDENCIES)
	pytest ./ -v -m "not mpi and not grass and not slow"

all: clean default grass optimization

grass:
	pytest ./test_grass.py -v -m grass

optimization:
	pytest ./test_optimization.py -v -m slow

mpi:
	srun -n 16 --mpi=pmi2 --qos=priority pytest ./test_mpi.py -m mpi

coverage:
	pytest ./ --cov=swimpy --cov-report=html -m "not mpi"
	rm -f .coverage

clean:
	git clean -f -x ./*
