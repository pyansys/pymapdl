"""gRPC specific class and methods for the MAPDL gRPC client """
import re
from warnings import warn
import shutil
import threading
import weakref
import io
import time
import logging
import os
import socket
from functools import wraps
import tempfile
import subprocess

import grpc
import numpy as np
from tqdm import tqdm
from grpc._channel import _InactiveRpcError, _MultiThreadedRendezvous
from ansys.grpc.mapdl import mapdl_pb2 as pb_types
from ansys.grpc.mapdl import mapdl_pb2_grpc as mapdl_grpc
from ansys.grpc.mapdl import ansys_kernel_pb2 as anskernel

from ansys.mapdl.core.mapdl import _MapdlCore
from ansys.mapdl.core.errors import MapdlExitedError, protect_grpc, MapdlRuntimeError
from ansys.mapdl.core.misc import supress_logging, run_as_prep7, last_created
from ansys.mapdl.core.post import PostProcessing
from ansys.mapdl.core.common_grpc import (parse_chunks,
                                          ANSYS_VALUE_TYPE,
                                          DEFAULT_CHUNKSIZE,
                                          DEFAULT_FILE_CHUNK_SIZE)
from ansys.mapdl.core import __version__, _LOCAL_PORTS

logger = logging.getLogger(__name__)

VOID_REQUEST = anskernel.EmptyRequest()


def chunk_raw(raw, save_as):
    with io.BytesIO(raw) as f:
        while True:
            piece = f.read(DEFAULT_FILE_CHUNK_SIZE)
            length = len(piece)
            if length == 0:
                return
            yield pb_types.UploadFileRequest(file_name=os.path.basename(save_as),
                                             chunk=anskernel.Chunk(payload=piece,
                                                                   size=length))


def get_file_chunks(filename, progress_bar=False):
    """Serializes a file into chunks"""
    pbar = None
    if progress_bar:
        n_bytes = os.path.getsize(filename)

        base_name = os.path.basename(filename)
        pbar = tqdm(total=n_bytes, desc='Uploading %s' % base_name,
                    unit='B', unit_scale=True, unit_divisor=1024)

    with open(filename, 'rb') as f:
        while True:
            piece = f.read(DEFAULT_FILE_CHUNK_SIZE)
            length = len(piece)
            if length == 0:
                if pbar is not None:
                    pbar.close()
                return

            if pbar is not None:
                pbar.update(length)

            chunk = anskernel.Chunk(payload=piece, size=length)
            yield pb_types.UploadFileRequest(file_name=os.path.basename(filename),
                                             chunk=chunk)


def save_chunks_to_file(chunks, filename, progress_bar=True,
                        file_size=None, target_name=''):
    """Saves chunks to a local file

    Returns
    -------
    file_size : int
        File size saved in bytes.  ``0`` means no file was written.
    """

    pbar = None
    if progress_bar:
        pbar = tqdm(total=file_size, desc='Downloading %s' % target_name,
                    unit='B', unit_scale=True, unit_divisor=1024)

    file_size = 0
    with open(filename, 'wb') as f:
        for chunk in chunks:
            f.write(chunk.payload)
            payload_size = len(chunk.payload)
            file_size += payload_size
            if pbar is not None:
                pbar.update(payload_size)

    if pbar is not None:
        pbar.close()

    return file_size


class RepeatingTimer(threading.Timer):
    """Run a function repeately"""
    def run(self):
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)


def check_valid_ip(ip):
    """Check for valid IP address"""
    if ip != 'localhost':
        ip = ip.replace('"', '').replace("'", '')
        socket.inet_aton(ip)


class MapdlGrpc(_MapdlCore):
    """This class connects to a GRPC MAPDL server and allows commands
    to be passed to a persistent session.

    Parameters
    ----------
    ip : str, optional
        IP address to connect to the server.  Defaults to 'localhost'.

    port : int, optional
        Port to connect to the mapdl server.  Defaults to 50052.

    timeout : float
        Maximum allowable time to connect to the MAPDL server.

    loglevel : str, optional
        Sets which messages are printed to the console.  Default
        'INFO' prints out all ANSYS messages, 'WARNING` prints only
        messages containing ANSYS warnings, and 'ERROR' prints only
        error messages.

    cleanup_on_exit : bool, optional
        Exit MAPDL when Python exits or when this instance is garbage
        collected.

    set_no_abort : bool, optional
        Sets MAPDL to not abort at the first error within /BATCH mode.
        Default ``True``.

    remove_temp_files : bool, optional
        Removes temporary files on exit if MAPDL is local.  Default
        ``False``.

    Examples
    --------
    Connect to an instance of MAPDL already running on locally on the
    default port 50052.

    >>> from ansys.mapdl import core as pymapdl
    >>> mapdl = pymapdl.Mapdl()

    Connect to an instance of MAPDL running on the LAN on a default port

    >>> mapdl = pymapdl.Mapdl('192.168.1.101')

    Connect to an instance of MAPDL running on the LAN on a non-default port

    >>> mapdl = pymapdl.Mapdl('192.168.1.101', port=60001)
    """

    def __init__(self, ip='127.0.0.1', port=None, timeout=15, loglevel='WARNING',
                 cleanup_on_exit=False, log_apdl=False, set_no_abort=True,
                 remove_temp_files=False, **kwargs):
        """Initialize connection to the mapdl server"""
        super().__init__(loglevel, **kwargs)

        check_valid_ip(ip)

        # gRPC request specific locks as these gRPC request are not thread safe
        self._vget_lock = False
        self._get_lock = False

        self._prioritize_thermal = False
        self._locked = False  # being used within MapdlPool
        self._stub = None
        self._cleanup = cleanup_on_exit
        self._remove_tmp = remove_temp_files
        self._jobname = kwargs.pop('jobname', 'file')
        self._path = kwargs.pop('run_location', None)
        self._busy = False  # used to check if running a command on the server
        self._channel_str = None
        self._local = ip in ['127.0.0.1', '127.0.1.1', 'localhost']
        self._ip = ip
        self._health_response_queue = None
        self._exiting = False

        if port is None:
            from ansys.mapdl.core.launcher import MAPDL_DEFAULT_PORT
            port = MAPDL_DEFAULT_PORT
        self._port = port
        self._server = None
        self._channel = None
        self._state = None
        self._stub = None
        self._timeout = timeout
        self._pids = []

        # try to connect over a series of attempts rather than one
        # single one.  This prevents a single failed connection from
        # blocking other attempts
        n_attempts = 30  # consider adding this as a kwarg
        connected = False
        attempt_timeout = timeout/n_attempts
        for i in range(n_attempts):
            self._log.debug('Connection attempt %d', i + 1)
            connected = self._connect(port, attempt_timeout, set_no_abort)
            if connected:
                break

        if not connected:
            raise IOError('Unable to connect to MAPDL gRPC instance at %s' %
                          self._channel_str)

        self._verify_local()
        if self._local:
            self._cache_pids()

    def _verify_local(self):
        """Check if Python is local to the MAPDL instance.

        Verify if python has assess to the MAPDL directory.
        """
        if self._local:
            if self._path is None:
                directory = self.directory
            else:
                directory = self._path

            if self._jobname is None:
                jobname = self.jobname
            else:
                jobname = self._jobname

            lockfile = os.path.join(directory, jobname + '.err')
            lockfile0 = os.path.join(directory, jobname + '0.err')
            if os.path.isfile(lockfile):
                return
            if os.path.isfile(lockfile0):
                return
            self._local = False

    def __repr__(self):
        info = super().__repr__()
        return info

    def _connect(self, port, timeout=5, set_no_abort=True, enable_health_check=False):
        """Establish a gRPC channel to a remote or local MAPDL instance.

        Parameters
        ----------
        timeout : float
            Time in seconds to wait until the connection has been established
        """
        self._server = {'ip': self._ip, 'port': port}
        self._channel_str = '%s:%d' % (self._ip, port)
        self._log.debug('Opening insecure channel at %s', self._channel_str)

        self._channel = grpc.insecure_channel(self._channel_str)
        self._state = grpc.channel_ready_future(self._channel)
        self._stub = mapdl_grpc.MapdlServiceStub(self._channel)

        # verify connection
        tstart = time.time()
        while ((time.time() - tstart) < timeout) and not self._state._matured:
            time.sleep(0.01)

        if not self._state._matured:  # pragma: no cover
            return False
        self._log.debug('Established connection to MAPDL gRPC')

        # keeps mapdl session alive
        self._timer = None
        if not self._local:
            self._initialised = threading.Event()
            self._t_trigger = time.time()
            self._t_delay = 30
            self._timer = threading.Thread(target=MapdlGrpc._threaded_heartbeat,
                                           args=(weakref.proxy(self), ))
            self._timer.daemon = True
            self._timer.start()

        # initialize mesh, post processing, and file explorer interfaces
        from ansys.mapdl.core.mesh_grpc import MeshGrpc
        from ansys.mapdl.core.xpl import ansXpl

        self._mesh_rep = MeshGrpc(self)
        self._post = PostProcessing(self)
        self._xpl = ansXpl(self)

        # TODO: version check
        
        request = anskernel.CtrlRequest(ctrl='VERSION')
        resp = self._stub.Ctrl( request)
        grpc_ver = resp.response.split('.')
        self._grpc_api_ver = float(grpc_ver[0]) + 0.1*float(grpc_ver[1])

        print('\ngRPC api release :    '  + str(self._grpc_api_ver) + '\n')

        # enable health check
        if enable_health_check:
            self._enable_health_check()

        # housekeeping otherwise, many failures in a row will cause
        # MAPDL to exit without returning anything useful.  Also
        # avoids abort in batch mode if set.
        if set_no_abort:
            self._set_no_abort()

        return True

    def _enable_health_check(self):
        """Places the status of the health check in _health_response_queue"""
        # lazy imports here to speed up module load
        from grpc_health.v1 import health_pb2, health_pb2_grpc

        def _consume_responses(response_iterator, response_queue):
            try:
                for response in response_iterator:
                    response_queue.put(response)
                # NOTE: we're doing absolutely nothing with this as
                # this point since the server side health check
                # doesn't change state.
            except Exception as err:
                if self._exiting:
                    return
                self._exited = True
                raise MapdlExitedError('Lost connection with MAPDL server') from None

        # enable health check
        from queue import Queue
        request = health_pb2.HealthCheckRequest()
        self._health_stub = health_pb2_grpc.HealthStub(self._channel)
        rendezvous = self._health_stub.Watch(request)

        # health check feature implemented after 2020R2
        try:
            status = rendezvous.next()
        except Exception as err:
            if err.code().name != 'UNIMPLEMENTED':
                raise err
            return

        if status.status != health_pb2.HealthCheckResponse.SERVING:
            raise MapdlRuntimeError('Unable to enable health check and/or connect to'
                                    ' the MAPDL server')

        self._health_response_queue = Queue()

        # allow main process to exit by setting daemon to true
        thread = threading.Thread(target=_consume_responses,
                                  args=(rendezvous, self._health_response_queue),
                                  daemon=True)
        thread.start()

    def _launch(self, start_parm):
        """Launch a local session of MAPDL in gRPC mode.

        This should only need to be used for legacy ``open_gui``
        """
        if not self._local:
            raise RuntimeError('Can only launch the GUI with a local instance of '
                               'MAPDL')
        from ansys.mapdl.core.launcher import launch_grpc
        self._exited = False  # reset exit state
        port, directory = launch_grpc(**start_parm)
        self._connect(port)

    @property
    def post_processing(self):
        """Post-process in an active MAPDL session.

        Examples
        --------
        Get the nodal displacement in the X direction for the first
        result set.

        >>> mapdl.set(1, 1)
        >>> disp_x = mapdl.post_processing.nodal_displacement('X')
        array([1.07512979e-04, 8.59137773e-05, 5.70690047e-05, ...,
               5.70333124e-05, 8.58600402e-05, 1.07445726e-04])
        """
        return self._post

    @supress_logging
    def _set_no_abort(self):
        """Do not abort MAPDL"""
        self.nerr(abort=-1)

    def _reset_cache(self):
        """Reset cached items"""
        self._mesh_rep._reset_cache()
        self._geometry._reset_cache()

    @property
    def _mesh(self):
        return self._mesh_rep

    def _run(self, cmd, verbose=False, mute=False):
        """Sens a command and return the response as a string.

        Parameters
        ----------
        cmd : str
            Valid MAPDL command.

        verbose : bool, optional
            Print the response of a command while it is being run.

        mute : bool, optional
            Request that no output be sent from the gRPC server.

        Examples
        --------
        Run a basic command.

        >>> mapdl.run('/PREP7')

        Run a command and suppress its output.

        >>> mapdl.run('/PREP7', mute=True)

        Run a command and stream its output while it is being run.

        >>> mapdl.run('/PREP7', mute=True)

        """
        if self._exited:
            raise MapdlExitedError

        # don't allow empty commands
        if not cmd.strip():
            raise ValueError('Empty commands not allowed')

        if len(cmd) > 639:  # CMD_MAX_LENGTH
            raise ValueError('Maximum command length must be less than 640 characters')

        self._busy = True
        if verbose:
            response = self._send_command_stream(cmd, True)
        else:
            response = self._send_command(cmd, mute=mute)
        self._busy = False
        return response.strip()

    @property
    def busy(self):
        """True when MAPDL gRPC server is executing a command"""
        return self._busy

    @protect_grpc
    def _send_command(self, cmd, mute=False):
        """Send a MAPDL command and return the response as a string"""
        opt = ''
        if mute:
            opt = 'MUTE'  # suppress any output

        request = pb_types.CmdRequest(command=cmd, opt=opt)
        # TODO: Capture keyboard exception and place this in a thread
        grpc_response = self._stub.SendCommand(request)

        resp = grpc_response.response
        if resp is not None:
            return resp.strip()
        return ''

    @protect_grpc
    def _send_command_stream(self, cmd, verbose=False):
        """Send a command and expect a streaming response"""
        request = pb_types.CmdRequest(command=cmd)
        metadata = [('time_step_stream', '100')]
        stream = self._stub.SendCommandS(request, metadata=metadata)
        response = []
        for item in stream:
            cmdout = '\n'.join(item.cmdout)
            if verbose:
                print(cmdout)
            response.append(cmdout.strip())

        return ''.join(response)

    def _threaded_heartbeat(self):
        """To be called from a thread to verify mapdl instance is alive"""
        self._initialised.set()
        while True:
            if self._exited:
                break

            try:
                time.sleep(self._t_delay)
                if not self.is_alive:
                    break
            except ReferenceError:
                break
            except Exception:
                continue

    def exit(self, save=False):
        """Exit MAPDL.

        Parameters
        ----------
        save : bool, optional
            Save the database on exit.  Default ``False``.

        Examples
        --------
        >>> mapdl.exit()
        """
        self._exiting = True
        self._log.debug('Exiting MAPDL')

        if save:
            try:
                self.save()
            except:
                pass

        self._kill()  # sets self._exited = True
        self._close_process()
        self._remove_lock_file()

        if self._remove_tmp and self._local:
            self._log.debug('Removing local temporary files')
            shutil.rmtree(self.directory, ignore_errors=True)

        if self._local and self._port in _LOCAL_PORTS:
            _LOCAL_PORTS.remove(self._port)

    def _kill(self):
        """Call exit(0) on the server."""
        self._ctrl('EXIT')
        self._exited = True

    def _close_process(self):
       """Close all MAPDL processes"""
       if self._local:
            for pid in self._pids:
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass

    def _cache_pids(self):
        """Store the process IDs used when launching MAPDL"""
        for filename in self.list_files():
            if 'cleanup' in filename:
                script = os.path.join(self.directory, filename)
                with open(script) as f:
                    raw = f.read()

                if os.name == 'nt':
                    pids = re.findall(r'/pid (\d+)', raw)
                else:
                    pids = set(re.findall(r'-9 (\d+)', raw))
                self._pids = [int(pid) for pid in pids]

    def _remove_lock_file(self):
        """Removes the lock file.

        Necessary to call this as a segfault of MAPDL or sys(0) will
        not remove the lock file.
        """
        mapdl_path = self.directory
        if mapdl_path:
            for lockname in [self.jobname + '.lock', 'file.lock']:
                lock_file = os.path.join(mapdl_path, lockname)
                if os.path.isfile(lock_file):
                    try:
                        os.remove(lock_file)
                    except OSError:
                        pass

    def _run_cleanup_script(self):  # pragma: no cover
        """Run the APDL cleanup script.

        On distributed runs MAPDL creates a cleanup script to kill the
        processes created by the ANSYS spawner.  Normally this file is
        removed when APDL exits normally, but on a failure, it's
        necessary to manually close these PIDs.
        """
        # run cleanup script when local
        if self._local:
            for filename in self.list_files():
                if 'cleanup' in filename:
                    script = os.path.join(self.directory, filename)
                    if not os.path.isfile(script):
                        return
                    if os.name != 'nt':
                        script = ['/bin/bash', script]
                    process = subprocess.Popen(script, shell=False,
                                               stdin=subprocess.PIPE,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE)
                    # always communicate to allow process to run
                    output, err = process.communicate()
                    self._log.debug('Cleanup output:\n\n%s\n%s', output.decode(),
                                    err.decode())

    def list_files(self):
        """List the files in the working directory of MAPDL.

        Returns
        -------
        files : list
            List of files in the working directory of MAPDL.

        Examples
        --------
        >>> files = mapdl.list_files()
        >>> for file in files: print(file)
        file.lock
        file0.bat
        file0.err
        file0.log
        file0.page
        file1.err
        file1.log
        file1.out
        file1.page
        """
        if self._local:  # simply return a python list of files
            local_path = self.directory
            if local_path:
                if os.path.isdir(local_path):
                    return os.listdir(self.directory)
            return []
        elif self._exited:
            raise RuntimeError('Cannot list remote files since MAPDL has exited')

        # this will sometimes return 'LINUX x6', 'LIN', or 'L'
        if 'L' in self.parameters.platform[:1]:
            cmd = 'ls'
        else:
            cmd = 'dir /b /a'

        files = self.sys(cmd).splitlines()
        if not files:
            warn('No files listed')
        return files

    @supress_logging
    def sys(self, cmd):
        """Pass a command string to the operating system.
        APDL Command: /SYS

        Parameters
        ----------
        cmd : str
            Command string, up to 639 characters (including blanks,
            commas, etc.). The specified string is passed verbatim to
            the operating system, i.e., no parameter substitution is
            performed.

        Returns
        -------
        output : str
            Output from the command.

        Examples
        --------
        >>> mapdl.sys('ls')

        Notes
        -----
        Passes a command string to the operating system for execution
        (see the Operations Guide).  Typical strings are system
        commands such as list, copy, rename, etc.  Control returns to
        the ANSYS program after the system procedure is completed.
        ANSYS may not be aware of your specific user environment. For
        example, on Linux this command may not recognize aliases,
        depending on the hardware platform and user environment.
        """
        # always redirect system output to a temporary file
        tmp_file = '__tmp_sys_out__'
        super().sys(f'{cmd} > {tmp_file}')
        if self._local:  # no need to download when local
            with open(os.path.join(self.directory, tmp_file)) as fobj:
                return fobj.read()
        return self._download_as_raw(tmp_file).decode()

    def download_result(self, path, progress_bar=False, preference=None):
        """Download remote result files to a local directory

        Examples
        --------
        Download remote result files into the current working directory

        >>> import os
        >>> mapdl.download_result(os.getcwd())

        """

        def _download(targets):
            for target in targets:
                save_name = os.path.join(path, target)
                self.download(target, save_name, progress_bar=progress_bar)

        if preference:
            if preference not in ['rst', 'rth']:
                raise ValueError("``preference`` must be either 'rst' or 'rth'")

        # result file basename is the jobname
        jobname = self.jobname
        rth_basename = '%s.%s' % (jobname, 'rth')
        rst_basename = '%s.%s' % (jobname, 'rst')

        remote_files = self.list_files()
        result_file = None

        if self._prioritize_thermal and rth_basename in remote_files:
            result_file = rth_basename
        elif rst_basename in remote_files and rth_basename in remote_files:
            if preference == 'rth':
                result_file = rth_basename
            else:
                result_file = rst_basename
        elif rst_basename in remote_files:
            result_file = rst_basename
        elif rth_basename in remote_files:
            result_file = rth_basename

        if result_file:  # found non-distributed result
            save_name = os.path.join(path, result_file)
            self.download(result_file, save_name, progress_bar=progress_bar)
            return save_name

        # otherwise, download all the distributed result files
        if jobname[-1].isnumeric():
            jobname += '_'

        rst_files = []
        rth_files = []
        for filename in remote_files:
            if 'rst' in filename and jobname in filename:
                rst_files.append(filename)
            elif 'rth' in filename and jobname in filename:
                rth_files.append(filename)

        if self._prioritize_thermal and rth_files:
            targets = rth_files
        else:
            if rst_files and rth_files:
                if preference is None:
                    raise ValueError('Found both structural and thermal results files.'
                                     '\nPlease specify which kind to download using:\n'
                                     '``preference="rth"`` or ``preference="rst"``')
                if preference == 'rst':
                    targets = rst_files
                elif preference == 'rth':
                    targets = rth_files
            elif rst_files:
                preference = 'rst'
                targets = rst_files
            elif rth_files:
                preference = 'rth'
                targets = rth_files
            else:
                remote_files_str = '\n'.join('\t%s' % item for item in remote_files)
                print('\t'.join('\n%s' % item for item in ['a', 'b', 'c']))
                raise FileNotFoundError('Unable to locate any result file from the '
                                        'following remote result files:\n\n'
                                        + remote_files_str)
        _download(targets)
        return os.path.join(path, jobname + '0.' + preference)

    @protect_grpc
    def _ctrl(self, cmd):
        """Issue control command to the mapdl server

        Available commands:

        - 'EXIT'
            Calls exit(0) on the server.

        - 'set_verb'
            Enables verbose mode on the server.

        Unavailable/Flaky:

        - 'time_stats'
            Prints a table for time stats on the server.
            This command appears to be disabled/broken.

        - 'mem-stats'
            To be added

        """
        self._log.debug('Issuing CtrlRequest "%s"', cmd)
        request = anskernel.CtrlRequest(ctrl=cmd)

        # handle socket closing upon exit
        if cmd.lower() == 'exit':
            try:
                # this always returns an error as the connection is closed
                self._stub.Ctrl(request)
            except (_InactiveRpcError, _MultiThreadedRendezvous):
                pass
            return

        # otherwise, simply send command
        self._stub.Ctrl(request)

    @wraps(_MapdlCore.cdread)
    def cdread(self, *args, **kwargs):
        """Wraps CDREAD"""
        option = kwargs.get('option', args[0])
        if option == 'ALL':
            raise ValueError('Option "ALL" not supported in gRPC mode.  Please '
                             'Input the geometry and mesh files separately '
                             r'with "\INPUT" or ``mapdl.input``')

        fname = kwargs.get('fname', args[1])
        kwargs.setdefault('verbose', False)
        kwargs.setdefault('progress_bar', False)
        self.input(fname, **kwargs)

    @protect_grpc
    def input(self, fname, verbose=False, progress_bar=False,
              time_step_stream=None, chunk_size=512, **kwargs):
        """Stream a local input file to a remote mapdl instance.
        Stream the response back and deserialize the output.
        Parameters
        ----------
        fname : str
            MAPDL input file to stream to the MAPDL grpc server.

        time_step_stream : int
            Time to wait between streaming updates to send back chunks
            from the listener file.  Larger values mean more data per
            chunk and less chunks, but if the command is short, will
            wait until time_step_stream is finished leading to a long
            execution time.

            Due to stability issues, the default time_step_stream is
            dependent on verbosity.  The defaults are:

            - ``verbose=True``: ``time_step_stream=500``
            - ``verbose=False``: ``time_step_stream=50``

            These defaults will be ignored if ``time_step_stream`` is
            manually set.

        Returns
        -------
        response : str
            Response from MAPDL.
        """
        # always check if file is present as the grpc and MAPDL errors
        # are unclear
        if self._local:
            if not os.path.isfile(fname):
                raise FileNotFoundError('Unable to locate filename "%s"' % fname)

            if not os.path.dirname(fname):
                filename = os.path.join(os.getcwd(), fname)
            else:
                filename = fname
        else:
            if not os.path.dirname(fname):
                # might be trying to run a local file.  Check if the
                # file exists remotely.
                if fname not in self.list_files():
                    self.upload(fname, progress_bar=progress_bar)
            else:
                self.upload(fname, progress_bar=progress_bar)
            filename = os.path.basename(fname)

        if time_step_stream is not None:
            if time_step_stream <= 0:
                raise ValueError('``time_step_stream`` must be greater than 0``')

        if verbose:
            if time_step_stream is None:
                time_step_stream = 500
            metadata = [('time_step_stream', str(time_step_stream)),
                        ('chunk_size', str(chunk_size))]

            request = pb_types.InputFileRequest(filename=filename)
            strouts = self._stub.InputFileS(request, metadata=metadata)
            responses = []
            for strout in strouts:
                lines = strout.cmdout
                # print out input as it is being run
                print('\n'.join(lines))
                responses.extend(lines)
            response = '\n'.join(responses)
            return response.strip()

        # otherwise, not verbose
        if time_step_stream is None:
            time_step_stream = 50
        metadata = [('time_step_stream', str(time_step_stream)),
                    ('chunk_size', str(chunk_size))]

        # since we can't directly run /INPUT, we have to write a
        # temporary input file that tells mainan to read the input
        # file.
        tmp_name = '_input_tmp_.inp'
        tmp_out = '_input_tmp_.out'
        tmp_dat = f"/OUT,{tmp_out}\n/INP,'{filename}'\n"
        if self._local:
            local_path = self.directory
            with open(os.path.join(local_path, tmp_name), 'w') as f:
                f.write(tmp_dat)
        else:
            self._upload_raw(tmp_dat.encode(), tmp_name)
        request = pb_types.InputFileRequest(filename=tmp_name)

        # even though we don't care about the output, we still
        # need to check.  otherwise, since inputfile is
        # non-blocking, we could corrupt the service
        chunks = self._stub.InputFileS(request, metadata=metadata)
        _ = [chunk.cmdout for chunk in chunks]  # unstable

        # all output (unless redirected) has been written to a temp output
        if self._local:
            with open(os.path.join(local_path, tmp_out)) as f:
                return f.read()

        # otherwise, read remote file
        return self._download_as_raw(tmp_out).decode('latin-1')

    def _flush_stored(self):
        """Writes stored commands to an input file and runs the input
        file.  Used with non_interactive.
        """
        self._log.debug('Flushing stored commands')

        commands = '\n'.join(self._stored_commands)
        if self._apdl_log:
            self._apdl_log.write(commands + '\n')

        self._log.debug('Writing the following commands to a temporary '
                        'apdl input file:\n%s', commands)

        # write to a temporary input file
        filename = os.path.join(tempfile.gettempdir(), '__tmp__.inp')
        with open(filename, 'w') as f:
            f.writelines(commands)

        self._store_commands = False
        self._stored_commands = []

        # run the stored commands
        out = self.input(filename, write_to_log=False, verbose=False,
                         chunk_size=DEFAULT_CHUNKSIZE, progress_bar=False)
        # skip the first line as it simply states that it's reading an input file
        self._response = out[out.find('LINE=       0') + 13:]
        self._log.info(self._response)

    @protect_grpc
    def _get(self, entity, entnum, item1, it1num, item2, it2num):
        """Sends gRPC *Get request.

        WARNING: Not thread SAFE.  Uses _get_lock to ensure multiple
        request aren't evaluated simultaneously.
        """
        cmd = ','.join([str(entity), str(entnum), str(item1),
                        str(it1num), str(item2), str(it2num)])

        # not threadsafe; don't allow multiple get commands
        while self._get_lock:
            time.sleep(0.001)

        self._get_lock = True
        try:
            getresponse = self._stub.Get(pb_types.GetRequest(getcmd=cmd))
        finally:
            self._get_lock = False

        if getresponse.type == 0:
            raise ValueError('This is either an invalid get request, or MAPDL is set'
                             ' to the wrong processor (e.g. on BEGIN LEVEL vs.'
                             ' POST26)')
        if getresponse.type == 1:
            return getresponse.dval
        elif getresponse.type == 2:
            return getresponse.sval

        raise RuntimeError('Unsupported type %s response from MAPDL' %
                           str(getresponse.type))

    @protect_grpc
    def download(self, target_name, out_file_name=None,
                 chunk_size=DEFAULT_CHUNKSIZE, progress_bar=True):
        """Download a file from the gRPC instance

        Parameters
        ----------
        target_name : str
            Target file on the server.  File must be in the same
            directory as the mapdl instance.  List current files with
            ``mapdl.list_files()``

        out_file_name : str, optional
            Save the filename as a different name other than the
            ``target_name``.

        chunk_size : int, optional
            Chunk size in bytes.  Must be less than 4MB.  Defaults to 256 kB.

        progress_bar : bool, optional Display a progress bar using
            ``tqdm`` when ``True``.  Helpful for showing download
            progress.

        Examples
        --------
        Download the remote result file "file.rst" as "my_result.rst"

        >>> mapdl.download('file.rst', 'my_result.rst')
        """
        if out_file_name is None:
            out_file_name = target_name

        request = pb_types.DownloadFileRequest(name=target_name)
        metadata = [('time_step_stream', '200'), ('chunk_size', str(chunk_size))]
        chunks = self._stub.DownloadFile(request, metadata=metadata)
        file_size = save_chunks_to_file(chunks, out_file_name,
                                        progress_bar=progress_bar,
                                        target_name=target_name)

        if not file_size:
            raise FileNotFoundError(f'File "{target_name}" is empty or does not exist')

    @protect_grpc
    def upload(self, file_name, progress_bar=True):
        """Upload a file to the grpc instance

        file_name : str
            Local file to upload.

        progress_bar : bool, optional Display a progress bar using
            ``tqdm`` when ``True``.  Helpful for showing download
            progress.

        Returns
        -------
        basename : str
            Base name of the file uploaded.  File can be accessed
            relative to the mapdl instance with this file name.

        Examples
        --------
        Upload "local_file.inp" while disabling the progress bar

        >>> mapdl.upload('local_file.inp', progress_bar=False)
        """
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f'Unable to locate filename {file_name}')

        chunks_generator = get_file_chunks(file_name, progress_bar=progress_bar)
        response = self._stub.UploadFile(chunks_generator)

        if not response.length:
            raise IOError('File failed to upload')
        return os.path.basename(file_name)

    @protect_grpc
    def _get_array(self, entity='', entnum='', item1='', it1num='', item2='',
                   it2num='', kloop='', **kwargs):
        """gRPC VGET request.

        Send a vget request, receive a bytes stream, and return it as
        a numpy array.

        Not thread safe as it uses a constant internal temporary
        parameter name.  This method uses _vget_lock to ensure
        multiple simultaneous request fail.

        Returns
        -------
        values : np.ndarray
            Numpy 1D array containing the requested *VGET item and entity.
        """
        if 'parm' in kwargs:
            raise ValueError('Parameter name `parm` not supported with gRPC')

        cmd = "%s,%s,%s,%s,%s,%s,%s" % (str(entity), str(entnum), str(item1),
                                        str(it1num), str(item2), str(it2num),
                                        str(kloop))

        while self._vget_lock:
            time.sleep(0.001)
        self._vget_lock = True
        try:
            chunks = self._stub.VGet2(pb_types.GetRequest(getcmd=cmd))
            values = parse_chunks(chunks)
        finally:
            self._vget_lock = False
        return values

    def _screenshot_path(self):
        """Returns the local path of the MAPDL generated screenshot.

        If necessary, downloads the remotely rendered file.
        """
        if self._local:
            return super()._screenshot_path()

        all_filenames = self.list_files()
        filenames = []
        for filename in all_filenames:
            if '.png' == filename[-4:]:
                filenames.append(filename)
        filenames.sort()
        filename = os.path.basename(filenames[-1])

        temp_dir = tempfile.gettempdir()
        save_name = os.path.join(temp_dir, 'tmp.png')
        self.download(filename, out_file_name=save_name)
        return save_name

    @protect_grpc
    def _download_as_raw(self, target_name):
        """Download a file from the gRPC instance as a binary
        string without saving it to disk.
        """
        request = pb_types.DownloadFileRequest(name=target_name)
        chunks = self._stub.DownloadFile(request)
        return b''.join([chunk.payload for chunk in chunks])

    @property
    def is_alive(self) -> bool:
        """True when there is an active connect to the gRPC server"""
        if self._exited:
            return False
        if self.busy:
            return True
        try:
            return bool(self.inquire('JOBNAME'))
        except:
            return False

    @property
    def xpl(self):
        """MAPDL file exploer

        Iteratively navigate through MAPDL files.

        Examples
        --------
        Read the MASS record from the "file.full" file

        >>> from ansys import Mapdl
        >>> mapdl = Mapdl()
        >>> xpl = mapdl.xpl
        >>> xpl.open('file.full')
        >>> vec = xpl.read('MASS')
        >>> vec.asarray()
        array([ 4,  7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52,
               55, 58,  1], dtype=int32)
        """
        return self._xpl

    @protect_grpc
    def scalar_param(self, pname):
        """Return a scalar parameter as a float"""
        request = pb_types.ParameterRequest(name=pname, array=False)
        presponse = self._stub.GetParameter(request)
        return float(presponse.val[0])

    @protect_grpc
    def _upload_raw(self, raw, save_as):  # consider private
        """Upload a binary string as a file"""
        chunks = chunk_raw(raw, save_as)
        response = self._stub.UploadFile(chunks)
        if response.length != len(raw):
            raise IOError('Raw Bytes failed to upload')

    ###########################################################################
    # not fully tested/implemented

    @protect_grpc
    def Param(self, pname):
        presponse = self._stub.GetParameter(pb_types.ParameterRequest(name=pname))
        return presponse.val

    @protect_grpc
    def Var(self, num):
        presponse = self._stub.GetVariable(pb_types.VariableRequest(inum=num))
        return presponse.val

    @property
    def math(self):
        """APDL math interface

        Examples
        --------
        Get the stiffness matrix from MAPDL

        >>> mm = mapdl.math.stiff()
        >>> matrix = k.asarray()
        <60x60 sparse matrix of type '<class 'numpy.float64'>'
            with 1734 stored elements in Compressed Sparse Row format>

        Get the mass matrix from mapdl

        >>> mm = mapdl.math.stiff()
        >>> matrix = k.asarray()
        <60x60 sparse matrix of type '<class 'numpy.float64'>'
            with 1734 stored elements in Compressed Sparse Row format>
        """
        from ansys.mapdl.core.math import MapdlMath
        return MapdlMath(self)

    @protect_grpc
    def _data_info(self, pname):
        """Returns the data type of a parameter

        APDLMATH vectors only.
        """
        request = pb_types.ParameterRequest(name=pname)
        return self._stub.GetDataInfo(request)

    @protect_grpc
    def _vec_data(self, pname):
        """Downloads vector data from a MAPDL MATH parameter"""
        dtype = ANSYS_VALUE_TYPE[self._data_info(pname).stype]
        request = pb_types.ParameterRequest(name=pname)
        chunks = self._stub.GetVecData(request)
        return parse_chunks(chunks, dtype)

    @protect_grpc
    def _mat_data(self, pname, raw=False):
        """Downloads matrix data from a parameter and returns a scipy sparse array"""
        try:
            from scipy import sparse
        except ImportError:  # pragma: no cover
            raise ImportError('Install ``scipy`` to use this feature')

        minfo = self._data_info(pname)
        stype = ANSYS_VALUE_TYPE[minfo.stype]
        mtype = minfo.objtype
        shape = (minfo.size1, minfo.size2)

        if mtype == 2:  # dense
            request = pb_types.ParameterRequest(name=pname)
            chunks = self._stub.GetMatData(request)
            values = parse_chunks(chunks, stype)
            return np.transpose(np.reshape(values, shape[::-1]))
        elif mtype == 3:
            indptr = self._vec_data(pname + "::ROWS")
            indices = self._vec_data(pname + "::COLS")
            vals = self._vec_data(pname + "::VALS")
            if raw:  # for debug
                return vals, indices, indptr, shape
            else:
                return sparse.csr_matrix((vals, indices, indptr), shape=shape)

        raise ValueError(f'Invalid matrix type "{mtype}"')

    @property
    def locked(self):
        """Instance is in use within a pool"""
        return self._locked

    @locked.setter
    def locked(self, new_value):
        self._locked = new_value

    @supress_logging
    def __str__(self):
        try:
            if self._exited:
                return 'MAPDL exited'
            stats = self.slashstatus('PROD')
        except:  # pragma: no cover
            return 'MAPDL exited'

        st = stats.find('*** Products ***')
        en = stats.find('*** PrePro')
        product = '\n'.join(stats[st:en].splitlines()[1:]).strip()

        # get product version
        stats = self.slashstatus('TITLE')
        st = stats.find('RELEASE')
        en = stats.find('INITIAL', st)
        mapdl_version = stats[st:en].split('CUSTOMER')[0].strip()

        info =  'Product:             %s\n' % product
        info += 'MAPDL Version:       %s\n' % mapdl_version
        info += 'ansys.mapdl Version: %s\n' % __version__

        return info

    @supress_logging
    @run_as_prep7
    def _generate_iges(self):
        """Save IGES geometry representation to disk"""
        basename = '_tmp.iges'
        if self._local:
            filename = os.path.join(self.directory, basename)
            self.igesout(basename, att=1)
        else:
            self.igesout(basename, att=1)
            filename = os.path.join(tempfile.gettempdir(), basename)
            self.download(basename, filename, progress_bar=False)
        return filename

    @property
    def _distributed_result_file(self):
        """Path of the distributed result file """
        if not self._distributed:
            return

        try:
            filename = self.inquire('RSTFILE')
            if not filename:
                filename = self.jobname
        except:
            filename = self.jobname

        # ansys decided that a jobname ended in a number needs a bonus "_"
        if filename[-1].isnumeric():
            filename += '_'

        rth_basename = '%s0.%s' % (filename, 'rth')
        rst_basename = '%s0.%s' % (filename, 'rst')

        rth_file = os.path.join(self.directory, rth_basename)
        rst_file = os.path.join(self.directory, rst_basename)

        if self._prioritize_thermal:
            if not os.path.isfile(rth_file):
                breakpoint()
                raise FileNotFoundError('Thermal Result not available')
            return rth_file

        if os.path.isfile(rth_file) and os.path.isfile(rst_file):
            return last_created([rth_file, rst_file])
        elif os.path.isfile(rth_file):
            return rth_file
        elif os.path.isfile(rst_file):
            return rst_file

    @property
    def _result_file(self):
        """Path of the non-distributed result file"""
        try:
            filename = self.inquire('RSTFILE')
            if not filename:
                filename = self.jobname
        except:
            filename = self.jobname

        try:
            ext = self.inquire('RSTEXT')
        except:  # check if rth file exists
            ext = ''

        if ext == '':
            rth_file = os.path.join(self.directory, '%s.%s' % (filename, 'rth'))
            rst_file = os.path.join(self.directory, '%s.%s' % (filename, 'rst'))

            if self._prioritize_thermal and os.path.isfile(rth_file):
                return rth_file

            if os.path.isfile(rth_file) and os.path.isfile(rst_file):
                return last_created([rth_file, rst_file])
            elif os.path.isfile(rth_file):
                return rth_file
            elif os.path.isfile(rst_file):
                return rst_file
        else:
            filename = os.path.join(self.directory, '%s.%s' % (filename, ext))
            if os.path.isfile(filename):
                return filename

    @property
    def thermal_result(self):
        """The thermal result object"""
        self._prioritize_thermal = True
        result = self.result
        self._prioritize_thermal = False
        return result

    def list_error_file(self):
        """Listing of errors written in JOBNAME.err"""
        files = self.list_files()
        jobname = self.jobname
        error_file = None
        for test_file in [f'{jobname}.err', f'{jobname}0.err']:
            if test_file in files:
                error_file = test_file
                break

        if not error_file:
            return None

        if self.local:
            return open(os.path.join(self.directory, error_file)).read()
        elif self._exited:
            raise MapdlExitedError('Cannot list error file when MAPDL Service has '
                                   'exited')

        return self._download_as_raw(error_file).decode('latin-1')

    @property
    def result(self):
        """Binary interface to the result file using ``pyansys.Result``

        Examples
        --------
        >>> mapdl.solve()
        >>> mapdl.finish()
        >>> result = mapdl.result
        >>> print(result)
        PyANSYS MAPDL Result file object
        Units       : User Defined
        Version     : 18.2
        Cyclic      : False
        Result Sets : 1
        Nodes       : 3083
        Elements    : 977

        Available Results:
        EMS : Miscellaneous summable items (normally includes face pressures)
        ENF : Nodal forces
        ENS : Nodal stresses
        ENG : Element energies and volume
        EEL : Nodal elastic strains
        ETH : Nodal thermal strains (includes swelling strains)
        EUL : Element euler angles
        EMN : Miscellaneous nonsummable items
        EPT : Nodal temperatures
        NSL : Nodal displacements
        RF  : Nodal reaction forces
        """
        from ansys.mapdl.reader import read_binary
        from ansys.mapdl.reader.rst import Result

        if not self._local:
            # download to temporary directory
            save_path = os.path.join(tempfile.gettempdir())
            result_path = self.download_result(save_path)
        else:
            if self._distributed_result_file and self._result_file:
                result_path = self._distributed_result_file
                result = Result(result_path, read_mesh=False)
                if result._is_cyclic:
                    result_path = self._result_file
                else:
                    # return the file with the last access time
                    filenames = [self._distributed_result_file, self._result_file]
                    result_path = last_created(filenames)
                    if result_path is None:  # if same return result_file
                        result_path = self._result_file

            elif self._distributed_result_file:
                result_path = self._distributed_result_file
                result = Result(result_path, read_mesh=False)
                if result._is_cyclic:
                    if not os.path.isfile(self._result_file):
                        raise RuntimeError('Distributed Cyclic result not supported')
                    result_path = self._result_file
            else:
                result_path = self._result_file

        if result_path is None:
            raise FileNotFoundError('No result file(s) at %s' % self.directory)
        if not os.path.isfile(result_path):
            raise FileNotFoundError('No results found at %s' % result_path)

        return read_binary(result_path)
