# **Cython Cheatsheet for Python and C Developers**

---

## **Setup**

1. **Install Cython**:

   ```bash
   pip install cython
   ```

2. **Create a Cython File**:
   Use `.pyx` extension for your Cython code.

   ```bash
   touch example.pyx
   ```

3. **Compile the Cython File**:
   Create a `setup.py` file:

   ```python
   from setuptools import setup
   from Cython.Build import cythonize

   setup(
       ext_modules=cythonize("example.pyx")
   )
   ```

   Build:

   ```bash
   python setup.py build_ext --inplace
   ```

---

## **Basic Syntax**

1. **Cython File Structure**:

   ```python
   # example.pyx

   def py_function():  # Regular Python function
       return "Hello, Python!"

   cpdef str cy_function():  # Python-callable C function
       return "Hello, Cython!"

   cdef int c_function():  # C function (not callable from Python)
       return 42
   ```

2. **C Declarations**:

   ```python
   cdef int a = 0          # Declare C integer
   cdef float b = 3.14     # Declare C float
   cdef char c = 'A'       # Declare C char
   ```

3. **Typing Variables**:

   ```python
   cdef int x = 10
   cdef float y = 5.5
   cdef str s = "Cython"
   ```

4. **Static Typing for Speed**:
   ```python
   def add_numbers(int x, int y):
       return x + y
   ```

---

## **Memory Views (Efficient Array Handling)**

1. **Using Cython with NumPy**:

   ```python
   import numpy as np
   cimport numpy as cnp

   def numpy_example(cnp.ndarray[cnp.float64_t, ndim=1] arr):
       cdef int i
       for i in range(arr.shape[0]):
           arr[i] *= 2
   ```

   Add NumPy in `setup.py`:

   ```python
   from setuptools import setup
   from Cython.Build import cythonize
   import numpy

   setup(
       ext_modules=cythonize("example.pyx"),
       include_dirs=[numpy.get_include()]
   )
   ```

2. **Memory View Syntax**:
   ```python
   cdef double[:, :] matrix = np.zeros((10, 10), dtype=np.float64)
   ```

---

## **Interfacing with C**

1. **Include C Headers**:

   ```python
   cdef extern from "math.h":
       double sin(double x)
       double cos(double x)

   def calculate_sin_cos(double x):
       return sin(x), cos(x)
   ```

2. **Declare C Structs**:

   ```python
   cdef struct Point:
       double x
       double y

   def create_point(double x, double y):
       cdef Point p
       p.x = x
       p.y = y
       return p
   ```

3. **Using C Functions in Python**:

   ```python
   cdef extern from "stdio.h":
       void printf(const char* format, ...)

   def print_message():
       printf(b"Hello from C!")
   ```

---

## **Optimizations**

1. **Disable Bounds Checking**:

   ```python
   # Disable for speed (use with caution!)
   @cython.boundscheck(False)
   def access_list_elements(int[:] lst):
       return lst[0]
   ```

2. **Disable Negative Indexing**:

   ```python
   @cython.wraparound(False)
   def access_list_elements(int[:] lst):
       return lst[-1]
   ```

3. **Declare Function Inline**:
   ```python
   cdef inline int add(int a, int b):
       return a + b
   ```

---

## **Special Features**

1. **Working with Python Objects**:

   ```python
   def mix_python_and_c():
       cdef int x = 42
       return x * 2  # Python interprets the result
   ```

2. **Multithreading**:

   ```python
   from cython.parallel import prange

   def parallel_sum(int[:] arr):
       cdef int i, total = 0
       for i in prange(len(arr), nogil=True):
           total += arr[i]
       return total
   ```

---

## **Build Tools**

1. **Use `pyximport` for Quick Prototyping**:

   ```python
   import pyximport
   pyximport.install()
   import example
   ```

2. **Use Jupyter Notebook with Cython**:
   Install `ipython` magic:
   ```bash
   pip install Cython
   ```
   In a notebook:
   ```python
   %load_ext Cython
   %%cython
   def cy_function():
       return "Hello from Cython!"
   ```

---

## **Debugging**

1. **Enable Line Tracing**:
   Add `annotate=True` to `cythonize` in `setup.py`:

   ```python
   cythonize("example.pyx", annotate=True)
   ```

   Open `example.html` to see Cython-to-C mappings.

2. **Debug with gdb**:
   Compile with debugging flags:
   ```python
   cythonize("example.pyx", gdb_debug=True)
   ```

---

This cheatsheet combines the basics and advanced features of Cython for efficient development! Let me know if you'd like more details on any section.
