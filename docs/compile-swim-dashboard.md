# Compile and run the SWIM dashboard app


## Compiling on the same OS as the target user
1) Install python or Anaconda
2) Make sure `pip`, `make` and `git` are running on the commandline
3) Make sure the `dependencies/*` are actually there, use `git submodule update --init --recursive` to bring all up-to-date
4) Optional: create a virtual python environment using `virtualenv` or `conda` and activate it
5) Run `make dist/swim-dashboard` to compile everything, the `dist/swim-dashboard` will include the Blankenstein catchment model and the swim and swim-dashboard executables
6) Zip `dist/swim-dashboard` and distribute it to users


## Using the app on a foreign machine
* the OS should be the same as the one it was compiled in with an OS version equal or new
* Under some OS the `swim-dashboard` and `swim` executables wont launch unless the user explicitly allows them to be executed in the system preferences

1) Unpack the `swim-dashboard.zip`
2) Execute the `swim-dashboard/swim-dashboard` executable and wait for it to load (can take 10-20sec)
3) Open a browser and navigate to `http://127.0.0.1:8054/`


## Troubleshooting
* Open a Terminal/Console/Command prompt (search for either of these in you OS) and navigate to the `swim-dashboard` directory (using `cd <path>`), then run `./swim-dashboard`
* This should produce some output and should finally tell you to navigate to the above link.