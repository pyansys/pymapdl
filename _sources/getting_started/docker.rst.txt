.. _docker:

**************************
Using MAPDL Through Docker
**************************
You can run MAPDL within a container on any OS using `docker` and
connect to it via PyMAPDL.

There are several situations in which it is advantageous to run MAPDL
in a containerized environment (e.g. Docker or singularity):

- Run in a consistent environment regardless of the host OS.
- Portability and ease of install.
- Large scale cluster deployment using Kubernetes
- Genuine application isolation through containerization.


Installing the MAPDL Image
--------------------------
There is a docker image hosted on the `PyMAPDL GitHub
<https://https://github.com/pyansys/pymapdl>`_ repository that you
can download using your GitHub credentials.

Assuming you have docker installed, you can get started by
authorizing docker to access this repository using a personal access
token.  Create a GH personal access token with ``packages read`` permissions
according to `Creating a personal access token <https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token>`_

Save that token to a file with:

.. code::

   echo XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX > GH_TOKEN.txt


This lets you send the token to docker without leaving the token value
in your history.  Next, authorize docker to access this repository
with:

.. code::

    GH_USERNAME=<my-github-username>
    cat GH_TOKEN.txt | docker login docker.pkg.github.com -u $GH_USERNAME --password-stdin


You can now launch MAPDL directly from docker with a short script or
directly from the command line.  Since this image contains no license
server, you will need to enter in your license server IP address the
``LICENSE_SERVER`` environment variable.  With that, you can launch
MAPDL with:

.. code::

    LICENSE_SERVER=1055@XXX.XXX.XXX.XXX
    VERSION=v21.1.0

    IMAGE=docker.pkg.github.com/pyansys/mapdl/mapdl_grpc:$VERSION
    docker run -e ANSYSLMD_LICENSE_FILE=$LICENSE_SERVER -p 50052:50052 $IMAGE -smp


Note that port `50052` (local to the container) is being mapped to
50052 on the host.  This makes it possible to launch several MAPDL
instances with different port mappings to allow for multiple instances
of MAPDL.

Once you've launched MAPDL you should see:

.. code::

    Start GRPC Server

    ##############################
    ### START GRPC SERVER      ###
    ##############################

    Server Executable   : MapdlGrpc Server
    Server listening on : 0.0.0.0:50052

Connecting to the MAPDL Container from Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can now connect to the instance with:

.. code:: python

    >>> from ansys.mapdl.core import Mapdl
    >>> mapdl = Mapdl()

If you mapped to any other port other than 50052, you should specify
that port when connecting to MAPDL with:

.. code:: python

    >>> mapdl = Mapdl(port=<my-port>)

Verify your connection with:

.. code:: python

    >>> print(mapdl)

    Product:             ANSYS Mechanical Enterprise
    MAPDL Version:       RELEASE  2021 R1           BUILD 21.0
    PyMAPDL Version:     Version: 0.57.0

Additional Considerations
-------------------------
In the command:

.. code::

    IMAGE=docker.pkg.github.com/pyansys/mapdl/mapdl_grpc:$VERSION
    docker run -e ANSYSLMD_LICENSE_FILE=$LICENSE_SERVER -p 50052:50052 $IMAGE -smp

You can provide additional command line parameters to MAPDL by simply
appending to the docker command.  For example, you can increase the
number of processors (up to the number available on the host machine)
with the `-np` switch.  For example:

.. code::

    IMAGE=docker.pkg.github.com/pyansys/mapdl/mapdl_grpc:$VERSION
    docker run -e ANSYSLMD_LICENSE_FILE=$LICENSE_SERVER -p 50052:50052 $IMAGE -np 4

For additional command line arguments please see the ansys
documentation at `ANSYS help <https://ansyshelp.ansys.com>`_.  Also,
be sure to have the appropriate license for additional HPC features.
