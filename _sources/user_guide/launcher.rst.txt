Initial Setup and Launching MAPDL Locally
-----------------------------------------
To run, ``ansys.mapdl.core`` needs to know the location of the MAPDL
binary.  Most of the time this can be automatically determined, but
non-standard installs will need to provide the location of MAPDL to
``ansys.mapdl``.  When running for the first time,
``ansys-mapdl-core`` will request the location of the MAPDL
executable.  You can test your installation of PyMAPDL and set it up
by running the following in python:

.. code:: python

    from ansys.mapdl import launch_mapdl
    mapdl = launch_mapdl()

Python will automatically attempt to detect your MAPDL binary based on
environmental variables.  If it is unable to find a copy of MAPDL, you
will be prompted for the location of the MAPDL executable.  Here is a
sample input for Linux and Windows:

.. code::

    Enter location of MAPDL executable: /usr/ansys_inc/v211/ansys/bin/ansys211

.. code::

    Enter location of MAPDL executable: C:\Program Files\ANSYS Inc\v211\ANSYS\bin\winx64\ansys211.exe

The settings file is stored locally and you will not not need to enter
the path again.  If you need to change the default ansys path
(i.e. changing the default version of MAPDL), run the following:

.. code:: python

    import ansys.mapdl.core as pymapdl
    new_path = 'C:\\Program Files\\ANSYS Inc\\v211\\ANSYS\\bin\\winx64\\ansys211.exe'
    pymapdl.change_default_ansys_path(new_path)


API Reference
~~~~~~~~~~~~~
For more details for controlling how MAPDL launches locally, see the
function description of ``launch_mapdl`` in the API reference at
:ref:`ref_launcher_api`.
