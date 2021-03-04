===============
Getting Started
===============
To use PyMAPDL, you need to have a local installation of Ansys.  The
version of ANSYS installed will dictate the interface and features
available to you.

Visit `Ansys <https://www.ansys.com/>`_ for more information on
getting a licensed copy of Ansys.


.. toctree::
   :hidden:
   :maxdepth: 2

   versioning
   docker


************
Installation
************

Python Module
~~~~~~~~~~~~~
The ``ansys.mapdl.core`` package currently supports Python 3.6 through
Python 3.8 on Windows, Mac OS, and Linux.

Install the latest release from `PyPi
<https://pypi.org/project/ansys-mapdl-core/>`_ with:

.. code::

   pip install ansys.mapdl-core

Alternatively, install the latest from `PyMAPDL GitHub
<https://github.com/pyansys/pymapdl/issues>`_ via:

.. code::

   pip install git+https://github.com/pyansys/pymapdl.git


For a local "development" version, install with:

.. code::

   git clone https://github.com/pyansys/pymapdl.git
   cd pymapdl
   pip install -e .

This will allow you to install the pymapdl ``ansys-mapdl-core`` module
and modify it locally and have the changes reflected in your setup
after restarting the Python kernel.


Offline Installation
~~~~~~~~~~~~~~~~~~~~
If you would like to install ``ansys-mapdl-core`` on a computer
without access to the internet, you can create a ``wheelhouse`` that
contains all the dependencies necessary to install a python package
without downloading each package individually.

On the host connected to the internet, run:

.. code::

   pip install wheel
   python -m pip wheel --wheel-dir=pyansys_wheelhouse ansys-mapdl-core

This creates a new directory called ``pyansys_wheelhouse`` which
contains every python package necessary to install
``ansys-mapdl-core``.

Next, zip the ``pyansys_wheelhouse`` directory and upload it to your
offline computer. On the offline computer, unzip it and then install
it with:

.. code::

   python -m pip install --no-index --find-links=pyansys_wheelhouse ansys-mapdl-core

This tells Python to install ``ansys-mapdl-core`` by only looking
within the ``pyansys_wheelhouse`` directory.

.. note::

   The OS and version of Python used to generate the wheelhouse must
   match the offline machine.


ANSYS Software Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~
For the latest features, you will need a copy of ANSYS 2021R1
installed locally, but PyMAPDL is compatible with ANSYS 17.0 and newer
on Windows and 13.0 on Linux.

.. note::

    The latest versions of ANSYS provide signifiantly better support
    and features.  Certain features will not be supported by earlier
    versions of ANSYS (e.g. APDL Math).


Verify Your Installation
~~~~~~~~~~~~~~~~~~~~~~~~
Check that you can start MAPDL from Python by running:

.. code:: python

    >>> from ansys.mapdl.core import launch_mapdl
    >>> mapdl = launch_mapdl()
    >>> print(mapdl)

    Product:             ANSYS Mechanical Enterprise
    MAPDL Version:       RELEASE  2021 R1           BUILD 21.0
    PyMAPDL Version:     Version: 0.57.0

If you see a response from the server, congratulations!  You're ready
to get started using MAPDL as a service.  For details regarding the
PyMAPDL interface, see :ref:`ref_mapdl_user_guide`.
