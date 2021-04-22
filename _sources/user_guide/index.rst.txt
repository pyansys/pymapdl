.. _ref_user_guide:

==========
User Guide
==========
This guide provides a general overview of the basics and usage of the
PyMAPDL module ``ansys-mapdl-core``.


..
   This toctreemust be a top level index to get it to show up in
   pydata_sphinx_theme

.. toctree::
   :maxdepth: 1
   :hidden:

   launcher
   mapdl
   mapdl_examples
   plotting
   mesh_geometry
   post
   parameters
   convert
   math
   pool
   xpl


PyMAPDL Basic Overview
======================
The ``launch_mapdl`` function within the ``ansys-mapdl-core`` module
creates an instance of of ``MAPDL`` in the background and sends
commands to that service.  Errors and warnings are processed
Pythonically letting the user develop a script real-time without
worrying about if it will function correctly when deployed in batch
mode.

MAPDL can be started from python in gRPC mode using
:func:`ansys.mapdl.launch_mapdl`.  This starts MAPDL in a temporary
directory by default.  You can change this to your current directory
with:

.. code:: python

    import os
    from ansys.mapdl.core import launch_mapdl

    path = os.getcwd()
    mapdl = launch_mapdl(run_location=path)

MAPDL is now active and you can send commands to it as a genuine a
Python class.  For example, if we wanted to create a surface using
keypoints we could run:

.. code:: python

    mapdl.run('/PREP7')
    mapdl.run('K, 1, 0, 0, 0')
    mapdl.run('K, 2, 1, 0, 0')
    mapdl.run('K, 3, 1, 1, 0')
    mapdl.run('K, 4, 0, 1, 0')
    mapdl.run('L, 1, 2')
    mapdl.run('L, 2, 3')
    mapdl.run('L, 3, 4')
    mapdl.run('L, 4, 1')
    mapdl.run('AL, 1, 2, 3, 4')

MAPDL interactively returns the result of each command and it is
stored to the logging module.  Errors are caught immediately.  For
example, if you input an invalid command:

.. code:: python

    >>> mapdl.run('AL, 1, 2, 3')

   MapdlRuntimeError: 
   AL, 1, 2, 3

   DEFINE AREA BY LIST OF LINES
   LINE LIST =     1    2    3
   (TRAVERSED IN SAME DIRECTION AS LINE     1)

   *** ERROR ***                           CP =       0.338   TIME= 09:45:36
   Keypoint 1 is referenced by only one line.  Improperly connected line   
   set for AL command.                                                     

This ``MapdlRuntimeError`` was caught immediately, and this means that
you can write your MAPDL scripts in python, run them interactively and
then as a batch without worrying if the script will run correctly if
you had instead outputted it to a script file.

The ``Mapdl`` class supports much more than just sending text to
MAPDL and includes higher level wrapping allowing for better scripting
and interaction with MAPDL.  See the :ref:`ref_example_gallery` for an
overview of the various advanced methods to visualize, script, and
interact with MAPDL.


Calling MAPDL Pythonically
~~~~~~~~~~~~~~~~~~~~~~~~~~
MAPDL functions can be called directly from an ``Mapdl`` instance in a
pythonic manner.  This is to simplify calling ANSYS, especially when
inputs are variables within Python.  For example, the following two
commands are equivalent:

.. code:: python

    mapdl.k(1, 0, 0, 0)
    mapdl.run('K, 1, 0, 0, 0')

This approach has some obvious advantages, chiefly that it's a easier
to script as ``ansys-mapdl-core`` takes care of the string formatting for you.
For example, inputting points from a numpy array:

.. code:: python

   # make 10 random keypoints in ANSYS
   points = np.random.random((10, 3))
   for i, (x, y, z) in enumerate(points):
       mapdl.k(i + 1, x, y, z)

Additionally, exceptions are caught and handled within Python.

.. code:: python

    >>> mapdl.run('AL, 1, 2, 3')

   Exception: 
   AL, 1, 2, 3

   DEFINE AREA BY LIST OF LINES
   LINE LIST =     1    2    3
   (TRAVERSED IN SAME DIRECTION AS LINE     1)

   *** ERROR ***                           CP =       0.338   TIME= 09:45:36
   Keypoint 1 is referenced by only one line.  Improperly connected line   
   set for AL command.                                                     


For longer scripts, instead of sending commands to MAPDL as in the
area creation example, we can instead run:

.. code:: python

    # clear existing geometry
    mapdl.finish()
    mapdl.clear()

    # create a square area using keypoints
    mapdl.prep7()
    mapdl.k(1, 0, 0, 0)
    mapdl.k(2, 1, 0, 0)
    mapdl.k(3, 1, 1, 0)
    mapdl.k(4, 0, 1, 0)    
    mapdl.l(1, 2)
    mapdl.l(2, 3)
    mapdl.l(3, 4)
    mapdl.l(4, 1)
    mapdl.al(1, 2, 3, 4)

This approach has some obvious advantages, chiefly that it's a bit
easier to script as ``MapdlGrpc`` takes care of the string formatting
for you.  For example, inputting points from a numpy array:

.. code:: python

   import numpy as np

   # make 10 random keypoints in MAPDL
   points = np.random.random((10, 3))
   for i, (x, y, z) in enumerate(points):
       mapdl.k(i + 1, x, y, z)

Additionally, each function with the MAPDL class has help associated
within it.  For example:

.. code:: python

    >>> help(mapdl.k)

    Help on method K in module ansys.mapdl.core.mapdl_grpc.MapdlGrpc:

    k(npt='', x='', y='', z='') method of ansys.mapdl.core.mapdl_grpc.MapdlGrpc
    instance

        APDL Command: K
        
        Defines a keypoint.
        
        Parameters
        ----------
        npt
            Reference number for keypoint.  If zero, the lowest available
            number is assigned [NUMSTR].
        
        x, y, z
            Keypoint location in the active coordinate system (may be R, θ, Z
            or R, θ, Φ).  If X = P, graphical picking is enabled and all other
            fields (including NPT) are ignored (valid only in the GUI).
        
        Notes
        -----
        Defines a keypoint in the active coordinate system [CSYS] for line,
        area, and volume descriptions.  A previously defined keypoint of the
        same number will be redefined.  Keypoints may be redefined only if it
        is not yet attached to a line or is not yet meshed.  Solid modeling in
        a toroidal system is not recommended.


Stability Considerations
------------------------
When connecting to a remote instance of MAPDL, there are some cases
where the MAPDL server will exit unexpectedly.  These issues are being
corrected and will be solved in 2021R2, but for the time being, there
are several ways to improve performance and stability of MADPL:

- When possible, pass ``mute=True`` to individual MAPDL commands or
  set it globally with ``mapdl.mute = True``.  This disables streaming
  back the response from MAPDL for each command and will marginally
  improve performance and stability.  Consider having a debug flag in
  your program or script so you can enable or disable logging and
  verbosity when needed.
- Avoid using input files if possible.  Attempt to ``upload`` nodes
  and elements and read them in via ``nread`` and ``eread``.  This is
  less than ideal and will be corrected in 2021R2.
