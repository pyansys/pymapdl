*********************************
Setting and Retrieving Parameters
*********************************
APDL parameters can be retrieved using ``ansys-mapdl-core`` using the
``Parameters`` class accessible from a ``mapdl`` instance.  For
example, if you wish to use the MAPDL ``GET`` command to populate a
parameter, you can then access the parameter with:

.. code:: python

   >>> from ansys.mapdl.core import launch_mapdl
   >>> mapdl = launch_mapdl()
   >>> mapdl.get('DEF_Y', 'NODE' , 2, 'U' ,'Y')
   >>> mapdl.parameters['DEF_Y']

You can also set both scalar and array parameters from python objects
using ``parameters`` with:

.. code:: python

   >>> mapdl.parameters['MY_ARRAY'] = np.arange(10000)
   >>> mapdl.parameters['MY_ARRAY']
   array([0.00000e+00, 1.00000e+00, 2.00000e+00, ..., 9.99997e+05,
          9.99998e+05, 9.99999e+05])

   >>> mapdl.parameters['MY_STRING'] = "helloworld"
   >>> mapdl.parameters['MY_STRING']
   "helloworld"

You an also access some built-in parameters normally accessed through
the ``\*GET`` command.  For example, instead of getting the current
routine with ``\*GET, ACTIVE, 0, ROUT``, you can access it with:

.. code:: python

  >>> mapdl.parameters.routine
  'Begin level'


For a full listing of the methods and attributes available to the ``Parameters`` class, please reference the :ref:`ref_parameters_api`.
