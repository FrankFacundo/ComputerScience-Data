#### Path to read code
https://github.com/numpy/numpy/issues/15567


#### Build
Numpy uses Cython to compile C.
https://numpy.org/doc/stable/user/building.html

```shell
python setup.py build_ext --inplace
```

####

Cython file types : 
https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#cython-file-types

There are three file types in Cython:
    - The implementation files, carrying a .py or .pyx suffix.
    - The definition files, carrying a .pxd suffix.
    - The include files, carrying a .pxi suffix.


In Python, what does "i" represent in .pyi extension?
    The i in .pyi stands for ‘interface’.

Ideas behind the code:
https://numpy.org/doc/stable/dev/internals.code-explanations.html

