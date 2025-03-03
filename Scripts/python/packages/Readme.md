# Creating and Publishing a Simple Python Library to PyPI

## Step 1: Set Up Your Project

Create a new directory for your library:

```sh
mkdir my_library
cd my_library
```

Inside this directory, create the main package folder:

```sh
mkdir my_library
```

Now, create an `__init__.py` file inside `my_library` to make it a package:

```sh
touch my_library/__init__.py
```

## Step 2: Write Your Library Code

Let's create a simple module inside `my_library`:

```python
# my_library/math_utils.py

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
```

Now, update `__init__.py` to import this module:

```python
# my_library/__init__.py
from .math_utils import add, subtract
```

## Step 3: Create `setup.py`

In the root directory (`my_library/`), create a `setup.py` file:

```python
from setuptools import setup, find_packages

setup(
    name="my_library",  # Choose a unique name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # Dependencies (if any)
    author="Your Name",
    author_email="your_email@example.com",
    description="A simple math library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_library",  # Your repository
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
```

## Step 4: Create a README File

Create a `README.md` file:

```md
# My Library

A simple Python library for basic math operations.

## Installation

```sh
pip install my_library
```

## Usage

```python
from my_library import add, subtract

print(add(5, 3))  # Output: 8
print(subtract(5, 3))  # Output: 2
```
```

## Step 5: Create a `pyproject.toml` File

Create a `pyproject.toml` file to specify build requirements:

```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
```

## Step 6: Build the Package

Ensure you have the required tools installed:

```sh
pip install setuptools wheel twine
```

Then, build your package:

```sh
python setup.py sdist bdist_wheel
```

## Step 7: Upload to PyPI

First, create an account on [PyPI](https://pypi.org/) and generate an API token.

Then, install Twine (if not already installed):

```sh
pip install twine
```

Upload your package to PyPI:

```sh
twine upload dist/*
```

You will be prompted to enter your PyPI username and password (or token).

## Step 8: Install and Test Your Package

Once uploaded, you can install your package from PyPI:

```sh
pip install my_library
```

Test it:

```python
from my_library import add, subtract
print(add(10, 2))  # Output: 12
print(subtract(10, 2))  # Output: 8
```

## Step 9: Versioning and Updates

To update your library, change the `version` in `setup.py`, rebuild, and upload again:

```sh
python setup.py sdist bdist_wheel
twine upload dist/*
```

## Conclusion

You've successfully created, packaged, and published a Python library to PyPI! 🎉

