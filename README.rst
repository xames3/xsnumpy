.. Author: Akshay Mestry <xa@mes3.dev>
.. Created on: Monday, November 18 2024
.. Last updated on: Monday, January 06 2025

===============================================================================
xsNumPy
===============================================================================

Etymology: *xs* (eXtra Small) and *numpy* (NumPy)

**xsNumPy** is personal pet-project of mine where I tried and implemented the
basic and bare-bones functionality of `NumPy`_ just using pure Python. This
project is a testament to the richness of NumPy's design. By reimplementing
its core features in a self-contained and minimalistic fashion, this project
aims to:

- Provide an educational tool for those seeking to understand array mechanics.
- Serve as a lightweight alternative for environments where dependencies
  must be minimized.
- Encourage developers to explore the intricacies of multidimensional
  array computation.

This project acknowledges the incredible contributions of the NumPy team and
community over decades of development. While this module reimagines NumPy's
functionality, it owes its design, inspiration, and motivation to the
pioneering work of the NumPy developers. This module is not a replacement for
NumPy but an homage to its brilliance and an opportunity to explore its
concepts from the ground up.

**xsNumPy** is a lightweight, pure-Python library inspired by NumPy, designed
to mimic essential array operations and features. This project is ideal for
learning and experimentation with multidimensional array processing and
numerical computing concepts.

-------------------------------------------------------------------------------
Installation
-------------------------------------------------------------------------------

.. See more at: https://stackoverflow.com/a/15268990

Install the latest version of xsNumPy using `pip`_:

.. code-block:: bash

    pip install -U git+https://github.com/xames3/xsnumpy.git#egg=xsnumpy

-------------------------------------------------------------------------------
Features
-------------------------------------------------------------------------------

As of now, **xsNumPy** offers the following features:

N-dimensional array (ndarray)
===============================================================================

- **xsnumpy.ndarray.** The central data structure representing N-dimensional
  arrays with support for:

  - Arbitrary shapes and data types.
  - Broadcasting\*\* for compatible operations (limited).
  - Arithmetic and comparison operations.

.. code-block:: python

    >>> import xsnumpy as xp
    >>>
    >>> a = xp.array([[1, 0], [0, 1]])
    >>> b = xp.array([[4, 1], [2, 2]])
    >>>
    >>> a + b
    array([[5, 1], 
           [2, 3]])
    >>> a - b
    array([[-3, -1], 
           [-2, -1]])
    >>> a * b
    array([[4, 0], 
           [0, 2]])
    >>> a / b
    array([[0.25, 0.  ], 
           [0.  ,  0.5]])
    >>> a // b
    array([[0, 0], 
           [0, 0]])
    >>> a ** b
    array([[1, 0], 
           [0, 1]])
    >>> a % b
    array([[1, 0], 
           [0, 1]])
    >>> a @ b
    array([[4, 1], 
           [2, 2]])
    >>> a < b
    array([[True, True], 
           [True, True]])
    >>> a >= b
    array([[False, False], 
           [False, False]])

Array creation routines
===============================================================================

- **xsnumpy.array.** Create an N-dimensional array.

.. code-block:: python

    >>> xp.array([[1, 2, 3], [4, 5, 6]])
    array([[1, 2, 3], 
           [4, 5, 6]])
    >>> xp.array([1, 2, 3.0])
    array([1. , 2. , 3. ])
    >>> xp.array([1, 2, 3], dtype=xp.bool)
    array([True, True, True])

- **xsnumpy.empty.** Create an uninitialized array of the given shape.

.. code-block:: python

    >>> xp.empty([2, 2])
    array([[0. , 0. ], 
           [0. , 0. ]])
    >>> xp.empty([2, 2], dtype=xp.int32)
    array([[0, 0], 
           [0, 0]])

- **xsnumpy.zeros.** Create an array filled with zeros.

.. code-block:: python

    >>> xp.zeros((2, 1))
    array([[0. ], 
           [0. ]])
    >>> xp.zeros((5,))
    array([0. , 0. , 0. , 0. , 0. ])
    >>> xp.zeros((5,), dtype=xp.int32)
    array([0, 0, 0, 0, 0])

- **xsnumpy.ones.** Create an array filled with ones.

.. code-block:: python

    >>> xp.ones((2, 1))
    array([[1. ], 
           [1. ]])
    >>> xp.ones((5,))
    array([1. , 1. , 1. , 1. , 1. ])

- **xsnumpy.full.** Create an array filled with *fill_value*.

.. code-block:: python

    >>> xp.full((2, 2), 10)
    array([[10. , 10. ], 
           [10. , 10. ]])

- **xsnumpy.arange.** Generate evenly spaced values within a given range.

.. code-block:: python

    >>> xp.arange(3)
    array([0, 1, 2])
    >>> xp.arange(3.0)
    array([0. , 1. , 2. ])
    >>> xp.arange(3, 7)
    array([3, 4, 5, 6])
    >>> xp.arange(3, 7, 2)
    array([3, 5])
    >>> xp.arange(0, 5, 0.5)
    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])

- **xsnumpy.eye.** Create a 2D array with ones on the diagonal and zeros
  elsewhere.

.. code-block:: python

    >>> xp.eye(2, dtype=xp.int32)
    array([[1, 0], 
           [0, 1]])

- **xsnumpy.identity.** Create an identity matrix or 2D array with ones on the
  main diagonal.

.. code-block:: python

    >>> xp.identity(3)
    array([[1. , 0. , 0. ], 
           [0. , 1. , 0. ], 
           [0. , 0. , 1. ]])

- **xsnumpy.tri.** Generate a lower triangular matrix filled with ones.

.. code-block:: python

    >>> xp.tri(3, 5, 2)
    array([[0. , 0. , 1. , 0. , 0. ], 
           [0. , 0. , 0. , 1. , 0. ], 
           [0. , 0. , 0. , 0. , 1. ]])
    >>> xp.tri(3, 5, -1, dtype=xp.int32)
    array([[0, 0, 0, 0, 0], 
           [1, 0, 0, 0, 0], 
           [0, 1, 0, 0, 0]])

- **xsnumpy.diag.** Extract a diagonal or construct a diagonal array.

.. code-block:: python

    >>> a = xp.arange(9).reshape((3, 3))
    >>> a
    array([[0, 1, 2], 
        [3, 4, 5], 
        [6, 7, 8]])
    >>> xp.diag(a)
    array([0, 4, 8])
    >>> xp.diag(a, k=1)
    array([1, 5])

Array attributes
===============================================================================

- **ndarray.shape.** Tuple of array dimensions.

.. code-block:: python

    >>> x = xp.array([1, 2, 3, 4])
    >>> x.shape
    (4,)
    >>> y = xp.zeros((2, 3, 4))
    >>> y.shape
    (2, 3, 4)
    >>> y.shape = (3, 8)
    >>> y
    array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ], 
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ], 
           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ]])

- **ndarray.strides.** Tuple of bytes to step in each dimension when traversing
  an array.

.. code-block:: python

    >>> y = xp.ones((2, 7))
    >>> y
    array([[1. , 1. , 1. , 1. , 1. , 1. , 1. ], 
           [1. , 1. , 1. , 1. , 1. , 1. , 1. ]])
    >>> y.strides
    (28, 4)

- **ndarray.ndim.** Number of array dimensions.

.. code-block:: python

    >>> x = xp.array([1, 2, 3])
    >>> x.ndim
    1
    >>> y = xp.zeros((2, 3, 4))
    >>> y.ndim
    3

- **ndarray.data.** Python buffer object pointing to the start of the array's
  data.

- **ndarray.size.** Number of elements in the array.

.. code-block:: python

    >>> x = xp.zeros((3, 5, 2))
    >>> x.size
    30

- **ndarray.itemsize.** Length of one array element in bytes.

.. code-block:: python

    >>> x = xp.array([1, 2, 3], dtype=xp.float64)
    >>> x.itemsize
    8
    >>> x = xp.array([1, 2, 3], dtype=xp.int16)
    >>> x.itemsize
    2

- **ndarray.nbytes.** Total bytes consumed by the elements of the array.

.. code-block:: python

    >>> x = xp.zeros((3, 5, 2), dtype=xp.float32)
    >>> x.nbytes
    120

- **ndarray.base.** Base object if memory is from some other object.

.. code-block:: python

    >>> x = xp.array([1, 2, 3, 4])
    >>> x.base is None
    True
    >>> y = x[2:]
    >>> y.base is x
    True

- **ndarray.dtype.** Data-type of the array's elements.

.. code-block:: python

    >>> x = xp.array([1, 2, 3, 4])
    >>> x.dtype
    dtype('int32')
    >>> type(x.dtype)
    <class 'xsnumpy.dtype'>

- **ndarray.T.** View of the transposed array.

.. code-block:: python

    >>> a = xp.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2], 
           [3, 4]])
    >>> a.T
    array([[1, 3], 
           [2, 4]])

Array methods
===============================================================================

- **ndarray.all().** Returns True if all elements evaluate to True.

.. code-block:: python

    >>> x = xp.array([[True, False], [True, True]])
    >>> x.all()
    False
    >>> x.all(axis=0)
    array([ True, False])
    >>> x = xp.array([-1, 4, 5])
    >>> x.all()
    True

- **ndarray.any().** Test whether any array element along a given axis
  evaluates to True.

.. code-block:: python

    >>> x = xp.array([[True, False], [True, True]])
    >>> x.any()
    True
    >>> x = xp.array([[True, False, True ], [False, False, False]])
    >>> x.any(axis=0)
    array([ True, False,  True])
    >>> a = xp.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]])
    >>> a.any(axis=0)
    array([ True, False,  True])

- **ndarray.astype().** Copies an array to a specified data type.

.. code-block:: python

    >>> arr = xp.array([1, 2, 3])
    >>> arr.astype(xp.float64)
    array([1. , 2. , 3. ], dtype=float64)

- **ndarray.fill().** Fill the array with a scalar value.

.. code-block:: python

    >>> a = xp.array([1, 2])
    >>> a.fill(0)
    >>> a
    array([0, 0])

- **ndarray.flatten().** Return a copy of the array collapsed into one
  dimension.

.. code-block:: python

    >>> a = xp.array([[1, 2], [3, 4]])
    >>> a.flatten()
    array([1, 2, 3, 4])

- **ndarray.item().** Copy an element of an array to a standard Python scalar
  and return it.

.. code-block:: python

    >>> x = xp.array([[2, 2, 6], [1, 3, 6], [1, 0, 1]])
    >>> x.item(3)
    1
    >>> x.item(7)
    0
    >>> x.item((0, 1))
    2
    >>> x.item((2, 2))
    1

- **ndarray.item().** Copy an element of an array to a standard Python scalar
  and return it.

.. code-block:: python

    >>> x = xp.array([[2, 2, 6], [1, 3, 6], [1, 0, 1]])
    >>> x.item(3)
    1
    >>> x.item(7)
    0
    >>> x.item((0, 1))
    2
    >>> x.item((2, 2))
    1

- **ndarray.min().** Return the minimum along a given axis.

.. code-block:: python

    >>> x = xp.array([[0, 1], [2, 3]])
    >>> x.min()
    0
    >>> x.min(axis=0)
    array([0, 1])
    >>> x.min(axis=1)
    array([0, 2])

- **ndarray.max().** Return the maximum along a given axis.

.. code-block:: python

    >>> x = xp.array([[0, 1], [2, 3]])
    >>> x.max()
    3
    >>> x.max(axis=0)
    array([2, 3])
    >>> x.max(axis=1)
    array([1, 3])

- **ndarray.sum().** Sum of array elements over a given axis.

.. code-block:: python

    >>> a = xp.array([0.5, 1.5])
    >>> a.sum()
    2.0
    >>> a = xp.array([[0, 1], [0, 5]])
    >>> a.sum()
    6
    >>> a.sum(axis=0)
    array([0, 6])
    >>> a.sum(axis=1)
    array([1, 5])

- **ndarray.prod().** Return the product of array elements over a given axis.

.. code-block:: python

    >>> a = xp.array([1., 2.])
    >>> a.prod()
    2.0
    >>> a = xp.array([[1., 2.], [3., 4.]])
    >>> a.prod()
    24.0
    >>> a.prod(axis=1)
    array([2.  , 12. ])
    >>> a.prod(axis=0)
    array([3. , 8. ])

- **ndarray.reshape().** Gives a new shape to an array without changing its
  data.

.. code-block:: python

    >>> a = xp.arange(6).reshape((3, 2))
    >>> a
    array([[0, 1], 
           [2, 3], 
           [4, 5]])
    >>> a = xp.array([[1, 2, 3], [4, 5, 6]])
    >>> a.reshape((6,))
    array([1, 2, 3, 4, 5, 6])

- **ndarray.transpose().** Returns an array with axes transposed.

.. code-block:: python

    >>> a = xp.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2], 
           [3, 4]])
    >>> a.transpose()
    array([[1, 3], 
           [2, 4]])
    >>> a = xp.array([1, 2, 3, 4])
    >>> a.transpose()
    array([1, 2, 3, 4])
    >>> a = xp.ones((1, 2, 3))
    >>> a.transpose((1, 0, 2)).shape
    (2, 1, 3)

Constants
===============================================================================

- **xsnumpy.e.** Euler's constant.

.. code-block:: python

    >>> xp.e
    2.718281828459045

- **xsnumpy.inf.** IEEE 754 floating point representation of (positive)
  infinity.

.. code-block:: python

    >>> xp.inf
    inf

- **xsnumpy.nan.** IEEE 754 floating point representation of Not a Number
  (NaN).

.. code-block:: python

    >>> xp.nan
    nan

- **xsnumpy.newaxis.** A convenient alias for None, useful for indexing arrays.

.. code-block:: python

    >>> xp.newaxis is None
    True

- **xsnumpy.pi.** Pi...

.. code-block:: python

    >>> xp.pi
    3.141592653589793

Linear algebra
===============================================================================

- **xsnumpy.dot.** Dot product of two arrays.

.. code-block:: python

    >>> xp.dot(3, 4)
    12
    >>> a = xp.array([[1, 0], [0, 1]])
    >>> b = xp.array([[4, 1], [2, 2]])
    >>> xp.dot(a, b)
    array([[4, 1], 
           [2, 2]])

- **xsnumpy.matmul.** Matrix multiplication product of two arrays.

.. code-block:: python

    >>> a = xp.array([[1, 0], [0, 1]])
    >>> b = xp.array([[4, 1], [2, 2]])
    >>> xp.matmul(a, b)
    array([[4, 1], 
           [2, 2]])

- **xsnumpy.add.** Add arguments element-wise.

.. code-block:: python

    >>> xp.add(3, 4)
    7
    >>> a = xp.array([[1, 0], [0, 1]])
    >>> b = xp.array([[4, 1], [2, 2]])
    >>> xp.add(a, b)
    array([[5. , 1. ], 
           [2. , 3. ]])

- **xsnumpy.subtract.** Subtract arguments element-wise.

.. code-block:: python

    >>> xp.subtract(3, 4)
    -1
    >>> a = xp.array([[1, 0], [0, 1]])
    >>> b = xp.array([[4, 1], [2, 2]])
    >>> xp.subtract(a, b)
    array([[-3. , -1. ], 
           [-2. , -1. ]])

- **xsnumpy.multiply.** Multiply arguments element-wise.

.. code-block:: python

    >>> xp.multiply(3, 4)
    12
    >>> a = xp.array([[1, 0], [0, 1]])
    >>> b = xp.array([[4, 1], [2, 2]])
    >>> xp.multiply(a, b)
    array([[4. , 0. ], 
           [0. , 2. ]])

- **xsnumpy.divide.** Divide arguments element-wise.

.. code-block:: python

    >>> xp.divide(4, 4)
    1.0
    >>> a = xp.array([[1, 0], [0, 1]])
    >>> b = xp.array([[4, 1], [2, 2]])
    >>> xp.divide(a, b)
    array([[0.25, 0.  ], 
           [0.  ,  0.5]])

- **xsnumpy.power.** First array elements raised to powers from second array,
  element-wise.

.. code-block:: python

    >>> xp.power(3, 4)
    81
    >>> a = xp.array([[1, 0], [0, 1]])
    >>> b = xp.array([[4, 1], [2, 2]])
    >>> xp.power(a, b)
    array([[1. , 0. ], 
           [0. , 1. ]])

Random Sampling
===============================================================================

- **xsnumpy.random.default_rng.** Construct a new Generator with the default
  BitGenerator (PCG64).

.. code-block:: python

    >>> rng = xp.random.default_rng(12345)
    >>> print(rng)
    Generator(PCG64)
    >>> rfloat = rng.random()
    >>> rfloat
    0.41661987254534116
    >>> type(rfloat)
    <class 'float'>
    >>> rints = rng.integers(low=0, high=10, size=3)
    >>> rints
    array([0, 4, 5])
    >>> type(rints[0])
    <class 'int'>
    >>> arr1 = rng.random((3, 3))
    >>> arr1
    array([[ 0.9317846894264221,   0.270244836807251,  0.4362284243106842], 
           [ 0.3730638325214386,  0.8741743564605713,  0.2610900104045868], 
           [ 0.6272147297859192,  0.6117693185806274, 0.18680904805660248]])

-------------------------------------------------------------------------------
Usage and Documentation
-------------------------------------------------------------------------------

The codebase is structured to be intuitive and mirrors the design principles
of NumPy to a significant extent. Comprehensive docstrings are provided for
each module and function, ensuring clarity and ease of understanding. Users
are encouraged to delve into the code, experiment with it, and modify it to
suit their learning curve.

Since, the implementation doesn't rely on any external package, it will work
with any CPython build v3.10 and higher. Technically, it should work on 3.9 and
below as well but due to some syntactical and type-aliasing changes, it might
not support. For instance, the typing module was significantly changed in
3.10, hence some features like `types.GenericAlias` and using native types
like `tuple`, `list`, etc. will not work. If you remove all the typing stuff,
the code will work just fine, at least that's what I hope.

**Note.** xsNumPy cannot and should not be used as an alternative to NumPy.
Another important note is the fact, this implementation doesn't fully support
array-broadcasting which is possibly one of the most important facet of NumPy.
Although, the existing features work with arrays when either their shapes
match or one of the array has ``n.dim`` is less than the other array.

-------------------------------------------------------------------------------
Contributions and Feedback
-------------------------------------------------------------------------------

Contributions to this project are warmly welcomed. Whether it's refining the
code, enhancing the documentation, or extending the current feature set, your
input is highly valued. Feedback, whether constructive criticism or 
commendation, is equally appreciated and will be instrumental in the evolution
of this educational tool.

-------------------------------------------------------------------------------
Acknowledgments
-------------------------------------------------------------------------------

This project is inspired by the remarkable work done by the `NumPy
Development Team`_. It is a tribute to their contributions to the field of
machine learning and the open-source community at large.

-------------------------------------------------------------------------------
License
-------------------------------------------------------------------------------

xsNumPy is licensed under the MIT License. See the `LICENSE`_ file for more
details.

.. _LICENSE: https://github.com/xames3/xsnumpy/blob/main/LICENSE
.. _NumPy Development Team: https://numpy.org/doc/stable/dev/index.html
.. _NumPy: https://numpy.org
.. _pip: https://pip.pypa.io/en/stable/getting-started/
