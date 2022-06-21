# Release Management

## Create a New Relase

### Increment Version

Open `swimpy/__init__.py` and increment the `__version__` value within the file.

### Release Birtual Environment

Create a virtual environment (if not already in one):
```
python -m venv release-env
source ./release-env/bin/activate
```

Install Dependencies:
```
pip install -e .
```

### Create Release artifacts

From the top directory, run:
```
make dist
```

This should create a `dist/` directory with similar contents:

```
$ tree dist/
dist/
├── swim
├── swimpy
├── swimpy-0.5.0-py2.py3-none-any.whl
└── swimpy-0.5.0.tar.gz
```

`swim` and `swimpy` are compiled, and will need to be compiled per platform (e.g. MacOS, Windows, Linux).

### Upload to Pypi sdist and wheel

```
make release
```

### Push docker images

Create Images:
```
make docker_build
```

Push Images
```
make docker_push
```

