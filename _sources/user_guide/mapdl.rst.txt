.. _ref_mapdl_user_guide:

**************************
PyMAPDL Language and Usage
**************************
This section gives you an overview of the PyMAPDL API for the
``Mapdl`` class.  For additional reference, see :ref:`ref_plotting_api`.

Overview
--------
When calling MAPDL commands as functions, each command has been
translated from its original MAPDL all CAPS format to a PEP8
compatible format.  For example, ``ESEL`` is now ``esel``.
Additionally, MAPDL commands containing a ``/`` or ``*`` have had
those characters removed, unless this causes a conflict with an
existing name.  Most notable is ``/SOLU`` which would conflict with
``SOLU``.  Therefore, the ``/SOLU`` has been renamed to ``slashsolu``
to differentiate it from ``solu``.  Out of the 1500 MAPDL commands,
about 15 start with ``slash`` and 8 with ``star``.  

MAPDL commands that normally have an empty space, such as ``ESEL, S,
TYPE, , 1`` should include an empty string when called by Python:

.. code:: python

    mapdl.esel('s', 'type', '', 1)

or these commands can be called using keyword arguments:

.. code:: python

    mapdl.esel('s', 'type', vmin=1)

None of these restrictions apply to commands run with ``run``, and it
may be easier to run some of these commands (e.g. "/SOLU"):

.. code:: python

    mapdl.run('/SOLU')
    mapdl.solve()

Some commands can only be run non-interactively from within in a
script.  PyMAPDL gets around this restriction by writing the commands
to a temporary input file and then reading the input file.  To run a
group of commands that must be run non-interactively, set the
``MAPDL`` object to run a series of commands as an input file by using
``non_interactive`` as in this example:

.. code:: python

    with mapdl.non_interactive:
        mapdl.run("*VWRITE,LABEL(1),VALUE(1,1),VALUE(1,2),VALUE(1,3)")
        mapdl.run("(1X,A8,'   ',F10.1,'  ',F10.1,'   ',1F5.3)")

Also note that macros created within PyMAPDL (rather than loaded from
a file) do not appear to run correctly.  For example, the macro
``DISP`` created using the ``*CREATE`` command within APDL:

.. code::

    ! SELECT NODES AT Z = 10 TO APPLY DISPLACEMENT
    *CREATE,DISP
    NSEL,R,LOC,Z,10
    D,ALL,UZ,ARG1
    NSEL,ALL
    /OUT,SCRATCH
    SOLVE
    *END

    ! Call the function
    *USE,DISP,-.032
    *USE,DISP,-.05
    *USE,DISP,-.1

Should be written as:

.. code:: python

    def DISP(ARG1='', ARG2='', ARG3='', ARG4='', ARG5='', ARG6='',
             ARG7='', ARG8='', ARG9='', ARG10='', ARG11='', ARG12='',
             ARG13='', ARG14='', ARG15='', ARG16='', ARG17='', ARG18=''):
        mapdl.nsel("R", "LOC", "Z", 10)  #SELECT NODES AT Z = 10 TO APPLY DISPLACEMENT
        mapdl.d("ALL", "UZ", ARG1)
        mapdl.nsel("ALL")
        mapdl.run("/OUT,SCRATCH")
        mapdl.solve()
    
    
    DISP(-.032)
    DISP(-.05)
    DISP(-.1)

If you have an existing input file with a macro, it can be converted
using the ``convert_script`` function and setting
``macros_as_functions=True``:

.. code:: python

    >>> from ansys.mapdl import core as pymapdl
    >>> pymapdl.convert_script(apdl_inputfile, pyscript, macros_as_functions=True)



Additional Options When Running Commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Commands can be run in ``mute`` or ``verbose`` mode, which allows you
to suppress or print the output in as it is being run for any MAPDL
command.  This can be especially helpful for long-running commands
like ``SOLVE``.  This works for the pythonic wrapping of all commands
and when using ``run``.

Run a command and suppress its output:

.. code:: python

    >>> mapdl.run('/PREP7', mute=True)
    >>> mapdl.prep7(mute=True)

Run a command and stream its output while it is being run.

.. code:: python

    >>> mapdl.run('SOLVE', mute=True)
    >>> mapdl.solve(verbose=True)

.. note::
    The ``verbose`` and ``mute`` features are only available when
    running MAPDL in gRPC mode.


Running Several Commands or an Input File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can run several MAPDL commands as a unified block.  This is useful
when using PyMAPDL with older MAPDL commands.  For example:

    >>> cmd = '''/prep7
    ! Mat
    MP,EX,1,200000
    MP,NUXY,1,0.3
    MP,DENS,1,7.85e-09
    ! Elements
    et,1,186
    et,2,154
    ! Geometry
    BLC4,0,0,1000,100,10
    ! Mesh
    esize,5
    vmesh,all
    nsel,s,loc,x,0
    d,all,all
    nsel,s,loc,x,999,1001
    type,2
    esurf
    esel,s,type,,2
    nsle
    sfe,all,3,pres,,-10
    allsel
    /solu
    antype,0
    solve
    /post1
    set,last
    plnsol,u,sum
    '''
    >>> resp = mapdl.run_multiline(cmd)
    >>> resp

     You have already entered the general preprocessor (PREP7).

     MATERIAL          1     EX   =   200000.0

     MATERIAL          1     NUXY =  0.3000000

     MATERIAL          1     DENS =  0.7850000E-08

     ELEMENT TYPE          1 IS SOLID186     3-D 20-NODE STRUCTURAL SOLID
      KEYOPT( 1- 6)=        0      0      0        0      0      0
      KEYOPT( 7-12)=        0      0      0        0      0      0
      KEYOPT(13-18)=        0      0      0        0      0      0

     CURRENT NODAL DOF SET IS  UX    UY    UZ
      THREE-DIMENSIONAL MODEL

     ELEMENT TYPE          2 IS SURF154      3-D STRUCTURAL SURFACE
      KEYOPT( 1- 6)=        0      0      0        0      0      0
      KEYOPT( 7-12)=        0      0      0        0      0      0
      KEYOPT(13-18)=        0      0      0        0      0      0

     CURRENT NODAL DOF SET IS  UX    UY    UZ
      THREE-DIMENSIONAL MODEL

     CREATE A HEXAHEDRAL VOLUME WITH
     X-DISTANCES FROM      0.000000000     TO      1000.000000
     Y-DISTANCES FROM      0.000000000     TO      100.0000000
     Z-DISTANCES FROM      0.000000000     TO      10.00000000

          OUTPUT VOLUME =     1

     DEFAULT ELEMENT DIVISIONS PER LINE BASED ON ELEMENT SIZE =   5.00

     GENERATE NODES AND ELEMENTS   IN  ALL  SELECTED VOLUMES
    
     NUMBER OF VOLUMES MESHED   =         1
     MAXIMUM NODE NUMBER        =     45765
     MAXIMUM ELEMENT NUMBER     =      8000

Alternatively, you can simply write the commands to a file and then
run it.  For example, if you have a ``"ds.dat"`` generated from Ansys
Workbench, you can run that with:

.. code:: python

    >>> resp = mapdl.input("ds.dat")


Conditional Statements and Loops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
APDL conditional statements such as ``*IF`` must be either implemented
pythonically or using ``with mapdl.non_interactive:``.   For example:

.. code:: 

    *IF,ARG1,EQ,0,THEN
      *GET,ARG4,NX,ARG2     ! RETRIEVE COORDINATE LOCATIONS OF BOTH NODES
      *GET,ARG5,NY,ARG2
      *GET,ARG6,NZ,ARG2
      *GET,ARG7,NX,ARG3
      *GET,ARG8,NY,ARG3
      *GET,ARG9,NZ,ARG3
    *ELSE
      *GET,ARG4,KX,ARG2     ! RETRIEVE COORDINATE LOCATIONS OF BOTH KEYPOINTS
      *GET,ARG5,KY,ARG2
      *GET,ARG6,KZ,ARG2
      *GET,ARG7,KX,ARG3
      *GET,ARG8,KY,ARG3
      *GET,ARG9,KZ,ARG3
    *ENDIF

Should be implemented as:

.. code:: python

    with mapdl.non_interactive:
        mapdl.run("*IF,ARG1,EQ,0,THEN")
        mapdl.run("*GET,ARG4,NX,ARG2     ")  # RETRIEVE COORDINATE LOCATIONS OF BOTH NODES
        mapdl.run("*GET,ARG5,NY,ARG2")
        mapdl.run("*GET,ARG6,NZ,ARG2")
        mapdl.run("*GET,ARG7,NX,ARG3")
        mapdl.run("*GET,ARG8,NY,ARG3")
        mapdl.run("*GET,ARG9,NZ,ARG3")
        mapdl.run("*ELSE")
        mapdl.run("*GET,ARG4,KX,ARG2     ")  # RETRIEVE COORDINATE LOCATIONS OF BOTH KEYPOINTS
        mapdl.run("*GET,ARG5,KY,ARG2")
        mapdl.run("*GET,ARG6,KZ,ARG2")
        mapdl.run("*GET,ARG7,KX,ARG3")
        mapdl.run("*GET,ARG8,KY,ARG3")
        mapdl.run("*GET,ARG9,KZ,ARG3")
        mapdl.run("*ENDIF")

Or pythonically as:

.. code:: python

    # MAPDL parameters can be obtained using load_parameters
    if ARG1 == 0:
        mapdl.run("*GET,ARG4,NX,ARG2     ")  # RETRIEVE COORDINATE LOCATIONS OF BOTH NODES
        mapdl.run("*GET,ARG5,NY,ARG2")
        mapdl.run("*GET,ARG6,NZ,ARG2")
        mapdl.run("*GET,ARG7,NX,ARG3")
        mapdl.run("*GET,ARG8,NY,ARG3")
        mapdl.run("*GET,ARG9,NZ,ARG3")
    else:
        mapdl.run("*GET,ARG4,KX,ARG2     ")  # RETRIEVE COORDINATE LOCATIONS OF BOTH KEYPOINTS
        mapdl.run("*GET,ARG5,KY,ARG2")
        mapdl.run("*GET,ARG6,KZ,ARG2")
        mapdl.run("*GET,ARG7,KX,ARG3")
        mapdl.run("*GET,ARG8,KY,ARG3")
        mapdl.run("*GET,ARG9,KZ,ARG3")

APDL loops using ``*DO`` or ``*DOWHILE`` should also be implemetned
using ``mapdl.non_interactive`` or pythonically.


Warnings and Errors
~~~~~~~~~~~~~~~~~~~
Errors are handled pythonically.  For example:

.. code:: python

    try:
        mapdl.solve()
    except:
        # do something else with MAPDL

Commands that are ignored within MAPDL are flagged as errors.  This is
different than MAPDL's default behavior where commands that are
ignored are treated as warnings.  For example, in ``ansys-mapdl-core``
running a command in the wrong session raises an error:

.. code:: python

    >>> mapdl.finish()
    >>> mapdl.k()

    Exception: 
    K, , , , 

     *** WARNING ***                         CP =       0.307   TIME= 11:05:01
     K is not a recognized BEGIN command, abbreviation, or macro.  This      
     command will be ignored.

You can change this behavior so ignored commands can be logged as warnings not raised as an exception by setting:

.. code:: python

   mapdl.allow_ignore = True
   mapdl.k()  # error ignored


Prompts
~~~~~~~
Prompts from MAPDL automatically continued as if MAPDL is in batch
mode.  Commands requiring user input, such as ``*VWRITE`` will fail
and must be entered in non-interactively.


APDL Command Logging
--------------------
While ``ansys-mapdl-core`` is designed to make it easier to control an
APDL session by calling it using Python, it may be necessary to call
MAPDL again using an input file generated from a PyMAPDL script.  This
is automatically enabled with the ``log_apdl='apdl.log'`` parameter.
Enabling this parameter will cause ``ansys-mapdl-core`` to write each
command run from a ``Mapdl`` object into a log file named
``"apdl.log"`` in the MAPDL working directory of the active ``mapdl``
object.  For example:

.. code:: python

    from ansys.mapdl.core import launch_mapdl

    ansys = launch_mapdl(log_apdl='apdl.log')
    ansys.prep7()
    ansys.k(1, 0, 0, 0)
    ansys.k(2, 1, 0, 0)
    ansys.k(3, 1, 1, 0)
    ansys.k(4, 0, 1, 0)    

Will write the following to ``"apdl.log"``

.. code::

    /PREP7,
    K,1,0,0,0
    K,2,1,0,0
    K,3,1,1,0
    K,4,0,1,0

This allows for the translation of a Python script to an APDL script
except for conditional statements, loops, or functions.


Interactive Breakpoint
----------------------
In most circumstances it is necessary or preferable to open up the
MAPDL GUI.  The ``ansys-mapdl-core`` module has an ``open_gui`` method
that allows you to seamlessly open up the GUI without losing work or
having to restart your session.  For example:

.. code:: python

    from ansys.mapdl.core import launch_mapdl
    mapdl = launch_mapdl()

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

    # open up the gui
    mapdl.open_gui()

    # it resumes where you left off...
    mapdl.et(1, 'MESH200', 6)
    mapdl.amesh('all')
    mapdl.eplot()    

This approach avoids the hassle of having to switch back and forth
between an interactive session and a scripting session.  Instead, you
can have one scripting session and open up a GUI from the scripting
session without losing work or progress.  Additionally, none of the
changes made in the GUI will affect the script.  You can experiment in
the GUI and the script will be left unaffected.


Running a Batch
---------------
Instead of running an MAPDL batch by calling MAPDL with an input file,
you can instead define a function that runs MAPDL.  This example runs
a mesh convergence study based on the maximum stress of a cylinder
with torsional loading.

.. code:: python

    import numpy as np
    from ansys.mapdl.core import launch_mapdl

    def cylinder_batch(elemsize, plot=False):
        """ Report the maximum von Mises stress of a Cantilever supported cylinder"""

        # clear
        mapdl.finish()
        mapdl.clear()

        # cylinder parameters
        radius = 2
        h_tip = 2
        height = 20
        force = 100/radius
        pressure = force/(h_tip*2*np.pi*radius)

        mapdl.prep7()
        mapdl.et(1, 186)
        mapdl.et(2, 154)
        mapdl.r(1)
        mapdl.r(2)

        # Aluminum properties (or something)
        mapdl.mp('ex', 1, 10e6)
        mapdl.mp('nuxy', 1, 0.3)
        mapdl.mp('dens', 1, 0.1/386.1)
        mapdl.mp('dens', 2, 0)

        # Simple cylinder
        for i in range(4):
            mapdl.cylind(radius, '', '', height, 90*(i-1), 90*i)

        mapdl.nummrg('kp')            

        # mesh cylinder
        mapdl.lsel('s', 'loc', 'x', 0)
        mapdl.lsel('r', 'loc', 'y', 0)
        mapdl.lsel('r', 'loc', 'z', 0, height - h_tip)
        # mapdl.lesize('all', elemsize*2)
        mapdl.mshape(0)
        mapdl.mshkey(1)
        mapdl.esize(elemsize)
        mapdl.allsel('all')
        mapdl.vsweep('ALL')
        mapdl.csys(1)
        mapdl.asel('s', 'loc', 'z', '', height - h_tip + 0.0001)
        mapdl.asel('r', 'loc', 'x', radius)
        mapdl.local(11, 1)
        mapdl.csys(0)
        mapdl.aatt(2, 2, 2, 11)
        mapdl.amesh('all')
        mapdl.finish()

        if plot:
            mapdl.view(1, 1, 1, 1)
            mapdl.eplot()

        # new solution
        mapdl.slashsolu()
        mapdl.antype('static', 'new')
        mapdl.eqslv('pcg', 1e-8)

        # Apply tangential pressure
        mapdl.esel('s', 'type', '', 2)
        mapdl.sfe('all', 2, 'pres', '', pressure)

        # Constrain bottom of cylinder/rod
        mapdl.asel('s', 'loc', 'z', 0)
        mapdl.nsla('s', 1)

        mapdl.d('all', 'all')
        mapdl.allsel()
        mapdl.psf('pres', '', 2)
        mapdl.pbc('u', 1)
        mapdl.solve()
        mapdl.finish()

        # access results using MAPDL object
        result = mapdl.result

        # to access the results you could have run:
        # from ansys.mapdl import reader as pymapdl_reader
        # resultfile = os.path.join(mapdl.path, '%s.rst' % mapdl.jobname)
        # result = pymapdl_reader.read_binary(result file)

        # Get maximum von Mises stress at result 1
        # Index 0 as it's zero based indexing
        nodenum, stress = result.principal_nodal_stress(0)

        # von Mises stress is the last column
        # must be nanmax as the shell element stress is not recorded
        maxstress = np.nanmax(stress[:, -1])

        # return number of nodes and max stress
        return nodenum.size, maxstress


    # initialize MAPDL
    mapdl = launch_mapdl(override=True, loglevel='ERROR')

    # call MAPDL to solve repeatedly
    result_summ = []
    for elemsize in np.linspace(0.6, 0.15, 15):
        # run the batch and report the results
        nnode, maxstress = cylinder_batch(elemsize, plot=False)
        result_summ.append([nnode, maxstress])
        print('Element size %f: %6d nodes and maximum vom Mises stress %f'
              % (elemsize, nnode, maxstress))

    # Exit MAPDL
    mapdl.exit()

This is the result from the script:

.. code::

    Element size 0.600000:   9657 nodes and maximum vom Mises stress 142.623505
    Element size 0.567857:  10213 nodes and maximum vom Mises stress 142.697800
    Element size 0.535714:  10769 nodes and maximum vom Mises stress 142.766510
    Element size 0.503571:  14177 nodes and maximum vom Mises stress 142.585388
    Element size 0.471429:  18371 nodes and maximum vom Mises stress 142.825684
    Element size 0.439286:  19724 nodes and maximum vom Mises stress 142.841202
    Element size 0.407143:  21412 nodes and maximum vom Mises stress 142.945984
    Element size 0.375000:  33502 nodes and maximum vom Mises stress 142.913437
    Element size 0.342857:  37877 nodes and maximum vom Mises stress 143.033401
    Element size 0.310714:  59432 nodes and maximum vom Mises stress 143.328842
    Element size 0.278571:  69106 nodes and maximum vom Mises stress 143.176086
    Element size 0.246429: 110547 nodes and maximum vom Mises stress 143.499329
    Element size 0.214286: 142496 nodes and maximum vom Mises stress 143.559128
    Element size 0.182143: 211966 nodes and maximum vom Mises stress 143.953430
    Element size 0.150000: 412324 nodes and maximum vom Mises stress 144.275406


Chaining Commands in MAPDL
--------------------------
MAPDL permits several commands on one line by using the separation
character ``"$"``.  This can be utilized within ``ansys-mapdl-core``
to effectively chain several commands together rather and send them to
MAPDL for execution rather than executing them individually.  This can
be helpful when you need to execute thousands of commands in a python
loop and don't need the individual results for each command.  For
example, if you wish to create a 1000 keypoints along the X axis you
would run:

.. code:: python

    xloc = np.linspace(0, 1, 1000)
    for x in xloc:
        mapdl.k(x=x)


However, since each command executes individually and returns a
response, it is much faster to send the commands to be executed by
MAPDL in groups and have ``ansys-mapdl-core`` handle grouping the
commands by running ``with mapdl.chain_commands``:

.. code:: python

    xloc = np.linspace(0, 1, 1000)
    with mapdl.chain_commands:
        for x in xloc:
            mapdl.k(x=x)

The execution time on this generally 4 to 10 times faster than running
each command individually.


Sending Arrays to MAPDL
-----------------------
You can send ``numpy`` arrays or Python lists directly to MAPDL using
``parameters['MY_ARRAY_NAME']``.  This is far more efficient than
individually sending parameters to MAPDL through Python with ``run``.
It uses ``*VREAD`` behind the scenes and will be replaced with a
faster interface in the future.

.. code:: python

    from ansys.mapdl.core import launch_mapdl
    import numpy as np
    mapdl = launch_mapdl()
    arr = np.random.random((5, 3))
    mapdl.parameters['MYARR'] = arr

Verify the data has been properly loaded to MAPDL by indexing
``mapdl.parameters`` as if it was a Python dictionary:

.. code:: python

   >>> array_from_mapdl = mapdl.parameters['MYARR']
   >>> array_from_mapdl
   array([[0.65516567, 0.96977939, 0.3224993 ],
          [0.58634927, 0.84392263, 0.18152529],
          [0.76719759, 0.45748876, 0.56432361],
          [0.78548338, 0.01042177, 0.57420062],
          [0.33189362, 0.9681039 , 0.47525875]])


Downloading a Remote MAPDL File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remote files can be listed and downloaded using ``ansys-mapdl-core``.
For example, to list the remote files and download one of them:

.. code:: python

    remote_files = mapdl.list_files()

    # ensure the result file is one of the remote files
    assert 'file.rst' in remote_files

    # download the remote result file
    mapdl.download('file.rst')

.. note::

   This is a gRPC feature only available in 2021R1 or newer.


Uploading a Local MAPDL File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can upload a local file a the remote mapdl instance with:

.. code:: python

    # upload a local file
    mapdl.upload('sample.db')

    # ensure the uploaded file is one of the remote files
    remote_files = mapdl.list_files()
    assert 'sample.db' in remote_files

.. note::

   This is a gRPC feature only available in 2021R1 or newer.
