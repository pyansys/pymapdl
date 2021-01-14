Create a Pool of MAPDL Instances
--------------------------------
The ``ansys.mapdl.core`` module contains the ``MapdlLocalPool`` class
to simplify creating multiple local instances of ``Mapdl`` for batch
processing.  This can be used for the batch processing of a set of
input files, convergence analysis, or other batch related processes.

To create the pool:

.. code:: python

    >>> from ansys.mapdl.core import LocalMapdlPool
    >>> pool = mapdl.LocalMapdlPool(10)
    'MAPDL Pool with 10 active instances'

You can also supply additional keyword arguments when creating the
pool.  For instance, create several instances with 1 CPU each running
at the current directory within their own isolated directories.

.. code:: python

    >>> import os
    >>> my_path = os.getcmd()
    >>> pool = mapdl.LocalMapdlPool(10, nproc=1, run_location=my_path)
    Creating Pool: 100%|████████| 10/10 [00:01<00:00,  1.43it/s]

Each individual instance of mapdl can be accessed with:

.. code:: python

    >>> pool[0]
    <ansys.mapdl.core.mapdl.MapdlGrpc object at 0x7f66270cc8d0>

Note that this is a self healing pool.  If an instance of MAPDL dies
during a batch process, that instance will be automatically restarted.
You can disable this behavior by setting ``restart_failed=False`` when
creating the pool.


Run a Set of Input Files
~~~~~~~~~~~~~~~~~~~~~~~~
You can use the pool to run a set of pre-generated input files using
``run_batch``.  For example, you can run the first set of 20
verification files with:

.. code:: python

    >>> from ansys.mapdl.core import examples
    >>> files = [examples.vmfiles['vm%d' % i] for i in range(1, 21)]
    >>> outputs = pool.run_batch(files)
    >>> len(outputs)
    20


Run a User Function
~~~~~~~~~~~~~~~~~~~
You can also use the pool to run a custom user function to run on each
instance of MAPDL over a set of inputs.  This example again uses set
of verification files as in the ``run_batch`` example, but implements
it as a function and outputs the final routine instead of the text
output from MAPDL.

.. code:: python

    >>> completed_indices = []
    >>> def func(mapdl, input_file, index):
            # input_file, index = args
            mapdl.clear()
            output = mapdl.input(input_file)
            completed_indices.append(index)
            return mapdl.parameters.routine
    >>> inputs = [(examples.vmfiles['vm%d' % i], i) for i in range(1, 10)]
    >>> output = pool.map(func, inputs, progress_bar=True, wait=True)
    ['Begin level',
     'Begin level',
     'Begin level',
     'Begin level',
     'Begin level',
     'Begin level',
     'Begin level',
     'Begin level',
     'Begin level']


API Description
~~~~~~~~~~~~~~~
For a full description of the PyMAPDL Pool API see :ref:`ref_pool_api`.
