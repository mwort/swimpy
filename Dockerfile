FROM mundialis/grass-py3-pdal:stable-ubuntu


# copy swimpy and dependencies
RUN mkdir -p /code/swimpy
WORKDIR /code/swimpy
COPY . .
RUN chmod -R a+rwx /code

# Make sure python3 is the default python
RUN rm -f /usr/bin/python && \
    ln -s $(which python3) /usr/bin/python

# Install swimpy
RUN pip3 install setuptools wheel && \
    pip3 install -U numpy ipython && \
    pip3 install -r requirements.txt && \
    pip3 install -U dependencies/modelmanager
RUN pip3 install -U /code/swimpy

# install swim
RUN apt-get update \
    && apt-get install -y --no-install-recommends --no-install-suggests \
        gfortran \
        libnetcdf-dev libnetcdff-dev \
    && apt-get autoremove \
    && apt-get clean

RUN make -C dependencies/swim/code clean && \
    make -C dependencies/swim/code && \
    mv dependencies/swim/code/swim /usr/bin

RUN make -C dependencies/swim/code clean && \
    make -C dependencies/swim/code CLIMNCDF=1 && \
    mv dependencies/swim/code/swim /usr/bin/swim-netcdf

# dedicated user
RUN useradd -m -U swim
USER swim

# Install m.swim (needs to be done after a new user is created)
RUN grass dependencies/m.swim/test/grassdb/utm32n/PERMANENT \
    --exec g.extension m.swim url=dependencies/m.swim

# run tests
RUN cd dependencies/swim/project && swim-netcdf ./
RUN make -C dependencies/m.swim/test
RUN make -C tests

# clean up
WORKDIR /data


