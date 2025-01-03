.. Author: Akshay Mestry <xa@mes3.dev>
.. Created on: Monday, November 18 2024
.. Last updated on: Friday, January 03 2025

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

1. A core data structure

- **ndarray.** The central data structure representing N-dimensional
  arrays with support for:

  - Arbitrary shapes and data types.
  - Broadcasting for compatible operations.

2. Array creation routines

- **array.** Create an N-dimensional array.

.. code-block:: python

    >>> import xsnumpy as xp
    >>> 
    >>> xp.array([[1, 2, 3], [4, 5, 6]])
    array([[1, 2, 3], 
           [4, 5, 6]])
    >>> xp.array([1, 2, 3.0])
    array([1. , 2. , 3. ])
    >>> xp.array([1, 2, 3], dtype=xp.bool)
    array([True, True, True])

- **empty.** Create an uninitialized array of the given shape.

.. code-block:: python

    >>> import xsnumpy as xp
    >>> 
    >>> xp.empty([2, 2])
    array([[0. , 0. ], 
           [0. , 0. ]])
    >>> xp.empty([2, 2], dtype=xp.int32)
    array([[0, 0], 
           [0, 0]])

- **zeros.** Create an array filled with zeros.

.. code-block:: python

    >>> xp.zeros((2, 1))
    array([[0. ], 
           [0. ]])
    >>> xp.zeros((5,))
    array([0. , 0. , 0. , 0. , 0. ])
    >>> xp.zeros((5,), dtype=xp.int32)
    array([0, 0, 0, 0, 0])

- **ones.** Create an array filled with ones.

.. code-block:: python

    >>> xp.ones((2, 1))
    array([[1. ], 
           [1. ]])
    >>> xp.ones((5,))
    array([1. , 1. , 1. , 1. , 1. ])

- **full.** Create an array filled with *fill_value*.

.. code-block:: python

    >>> xp.full((2, 2), 10)
    array([[10. , 10. ], 
           [10. , 10. ]])

- **arange.** Generate evenly spaced values within a given range.

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

- **eye.** Create a 2D array with ones on the diagonal and zeros elsewhere.

.. code-block:: python

    >>> xp.eye(2, dtype=xp.int32)
    array([[1, 0], 
           [0, 1]])

- **identity.** Create an identity matrix or 2D array with ones on the main
  diagonal.

.. code-block:: python

    >>> xp.identity(3)
    array([[1. , 0. , 0. ], 
           [0. , 1. , 0. ], 
           [0. , 0. , 1. ]])

- **tri.** Generate a lower triangular matrix filled with ones.

.. code-block:: python

    >>> xp.tri(3, 5, 2)
    array([[0. , 0. , 1. , 0. , 0. ], 
           [0. , 0. , 0. , 1. , 0. ], 
           [0. , 0. , 0. , 0. , 1. ]])
    >>> xp.tri(3, 5, -1, dtype=xp.int32)
    array([[0, 0, 0, 0, 0], 
           [1, 0, 0, 0, 0], 
           [0, 1, 0, 0, 0]])

-------------------------------------------------------------------------------
Usage and Documentation
-------------------------------------------------------------------------------

The codebase is structured to be intuitive and mirrors the design principles
of NumPy to a significant extent. Comprehensive docstrings are provided for
each module and function, ensuring clarity and ease of understanding. Users
are encouraged to delve into the code, experiment with it, and modify it to
suit their learning curve.

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
development team`_. It is a tribute to their contributions to the field of
machine learning and the open-source community at large.

.. _NumPy: https://numpy.org
.. _NumPy development team: https://numpy.org/doc/stable/dev/index.html
.. _pip: https://pip.pypa.io/en/stable/getting-started/
