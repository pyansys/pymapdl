*************************
Versioning and Interfaces
*************************
The PyMAPDL project attempts to maintain compatibility with legacy
versions of MAPDL while allowing for support of faster and better
interfaces with the latest versions of MAPDL.


ANSYS 2021R1 and Newer
~~~~~~~~~~~~~~~~~~~~~~
ANSYS v2020R1 and newer support the latest gRPC interface, allowing
for remote management of MAPDL with rapid streaming of mesh, results,
and files from the MAPDL service.  Should you have the applicable
license, you can even install and use MAPDL with docker, enabling you
to run and solve even on officially unsupported platforms like Mac
OS.


ANSYS 17.0 to 2020R1 - CORBA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ANSYS 17.0 supports the legacy CORBA interface, enabled with the
``ansys.mapdl.corba`` module.  This interface allows you to send only
text to and from the MAPDL service, relying on file IO for all other
operations.  While not as performant as gRPC, this interface still
allows you to control a local instance of MAPDL.  These versions of
MAPDL support specific versions of Windows and Linux.  See `Ansys Platform Support
<https://www.ansys.com/solutions/solutions-by-role/it-professionals/platform-support>`_
for more details on the supported platforms.

.. Note::

   The CORBA interface will likely be phased out from MAPDL at some
   point.  The gRPC interface is faster, more stable, and can run in
   both local and remote connection configurations.


Earlier than ANSYS 17.0
~~~~~~~~~~~~~~~~~~~~~~~
The ``PyMAPDL`` project supports up to ANSYS v13.0 on Linux using a
console interface.  Like CORBA, it allows for the exchange of text to
and from the ANSYS instance, but unlike the CORBA

.. Warning::

   Console specific support will be depreciated at some point and it
   is recommended to shift to a modern version of Ansys to continue to
   use PyMAPDL.
