# Installing Python

## Windows

### Windows Store

You can open install python through the windows store.
This is also the default when running `python` in the terminal, which
will re-direct you to the windows store if python isn't currently installed.

Search for "python" in the windows store and and you should be able to find
a recent version of python.

However, this is just a base installation of python, you will also need to download
pip; which is the most commonly used package manager for python.
```
py -m ensurepip --default-pip
```

A full guide on install pip [is available in this installation guide](https://packaging.python.org/en/latest/tutorials/installing-packages/).

### Official Python Installer

The [official python releases for windows](https://www.python.org/downloads/windows/)
will have many options for installation. The easist of which will be "Windows installer"
which will be a guided installer.

Depending on how your system's PATH is configured, the `python` command may still default
to the `python` command stub which references the windows store.

### Anaconda

Anaconda is a popular python distribution platform, especially amongst the science community.
Their [official installer page](https://www.anaconda.com/products/distribution) will provide
you with an installer and many other necesities such as pip.

## MacOS

MacOS comes pre-installed with python and pip. One note is that `python` still defaults to
python2 on MacOS. So explicit usage of `python3` is required to select python 3.

## Linux

`python3` is likely to be installed on most linux platforms as it's a commonly bundled tool with most
distributions. Please refer to your respective distribution's documentation on how to install python.
