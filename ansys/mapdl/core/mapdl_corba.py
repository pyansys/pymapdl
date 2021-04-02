"""CORBA implementation of the MAPDL interface"""
import atexit
import subprocess
import time
import re
import os
import weakref

from ansys.mapdl.core.mapdl import _MapdlCore
from ansys.mapdl.core.misc import threaded, random_string
from ansys.mapdl.core.errors import MapdlRuntimeError, MapdlExitedError

from ansys_corba import CORBA

INSTANCES = []

@atexit.register
def _cleanup():
    for instance in INSTANCES:
        try:
            if instance() is not None:
                instance().exit()
        except:
            pass


def tail(filename, nlines):
    """Read the last nlines of a text file """
    with open(filename) as qfile:
        qfile.seek(0, os.SEEK_END)
        endf = position = qfile.tell()
        linecnt = 0
        while position >= 0:
            qfile.seek(position)
            next_char = qfile.read(1)
            if next_char == "\n" and position != endf-1:
                linecnt += 1

            if linecnt == nlines:
                break
            position -= 1

        if position < 0:
            qfile.seek(0)

        return qfile.read()


def launch_corba(exec_file=None, run_location=None, jobname=None, nproc=None,
                 verbose=False, additional_switches='', start_timeout=60):
    """Start MAPDL in AAS mode

    Notes
    -----
    The CORBA interface is likely to fail on computers with multiple
    network adapters.  The ANSYS RPC isn't smart enough to determine
    the right adapter and will likely try to communicate on the wrong
    IP.
    """
    # Using stored parameters so launch command can be run from a
    # cached state (when launching the GUI)

    # can't run /BATCH in windows, so we trick it using "-b" and
    # provide a dummy input file
    if os.name == 'nt':
        # must be a random filename to avoid conflicts with other
        # potential instances
        tmp_file = '%s.inp' % random_string(10)
        with open(os.path.join(run_location, tmp_file), 'w') as f:
            f.write('FINISH')
        additional_switches += ' -b -i %s -o out.txt' % tmp_file

    # command must include "aas" flag to start MAPDL server
    command = '"%s" -aas -j %s -np %d %s' % (exec_file,
                                             jobname,
                                             nproc,
                                             additional_switches)

    # remove any broadcast files
    broadcast_file = os.path.join(run_location, 'mapdl_broadcasts.txt')
    if os.path.isfile(broadcast_file):
        os.remove(broadcast_file)

    if verbose:
        subprocess.Popen(command, shell=True,
                         cwd=run_location)
    else:
        subprocess.Popen(command, shell=True,
                         cwd=run_location,
                         stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)

    # listen for broadcast file
    telapsed = 0
    tstart = time.time()
    started_rpc = False
    while telapsed < start_timeout and not started_rpc:
        try:
            if os.path.isfile(broadcast_file):
                broadcast = open(broadcast_file).read()
                # see if connection to RPC has been made
                rpc_txt = 'visited:collaborativecosolverunitior-set:'
                started_rpc = rpc_txt in broadcast

            time.sleep(0.1)
            telapsed = time.time() - tstart

        except KeyboardInterrupt:
            raise KeyboardInterrupt

    # exit if timed out
    if not started_rpc:
        err_str = 'Unable to start ANSYS within %.1f seconds' % start_timeout
        if os.path.isfile(broadcast_file):
            broadcast = open(broadcast_file).read()
            err_str += '\n\nLast broadcast:\n%s' % broadcast
        raise TimeoutError(err_str)

    # return CORBA key
    keyfile = os.path.join(run_location, 'aaS_MapdlId.txt')
    return open(keyfile).read()


class MapdlCorba(_MapdlCore):
    """CORBA implementation of the MAPDL interface

    Parameters
    ----------
    corba_key : str
        CORBA key used to start the corba interface

    start : dict
        Additional start parameters to be passed to launcher when
        launching the gui interactively.

        exec_file, run_location, jobname='file', nproc=2,
        additional_switches='', timeout

    verbose : bool
        Print all output from MAPDL to Python.  Useful for debugging

    """

    def __init__(self, loglevel='INFO', log_apdl='w', use_vtk=True,
                 log_broadcast=False, verbose=False, **start_parm):
        """Open a connection to MAPDL via a CORBA interface"""
        super().__init__(loglevel=loglevel, use_vtk=use_vtk, log_apdl=log_apdl,
                         log_broadcast=False, **start_parm)
        self._broadcast_logger = None
        self._server = None
        self._outfile = None
        self._log_broadcast = log_broadcast
        self._launch(start_parm, verbose)
        super().__init__(loglevel=loglevel, use_vtk=use_vtk, log_apdl=log_apdl,
                         **start_parm)

        # critical for collection
        INSTANCES.append(weakref.ref(self))

    def _launch(self, start_parm, verbose=False):
        """Launch a CORBA instance using the MAPDL AAS flag"""
        corba_key = launch_corba(verbose=verbose, **start_parm)

        orb = CORBA.ORB_init()
        self._server = orb.string_to_object(corba_key)

        # verify you can connect to MAPDL
        try:
            self._server.getComponentName()
        except:
            raise MapdlRuntimeError('Unable to connect to APDL server')

        # must set to non-interactive in linux
        if os.name == 'posix':
            self.batch()

        self._log.debug('Connected to ANSYS using CORBA interface with key %s',
                        corba_key)

        # separate logger for broadcast file
        if self._log_broadcast:
            self._broadcast_logger = self._start_broadcast_logger()

    @property
    def _broadcast_file(self):
        return os.path.join(self.directory, 'mapdl_broadcasts.txt')

    @threaded
    def _start_broadcast_logger(self, update_rate=1.0):
        """Separate logger using broadcast_file """
        # listen to broadcast file
        loadstep = 0
        overall_progress = 0
        try:
            old_tail = ''
            old_size = 0
            while not self._exited:
                new_size = os.path.getsize(self._broadcast_file)
                if new_size != old_size:
                    old_size = new_size
                    new_tail = tail(self._broadcast_file, 4)
                    if new_tail != old_tail:
                        lines = new_tail.split('>>')
                        for line in lines:
                            line = line.strip().replace('<<broadcast::', '')
                            if "current-load-step" in line:
                                n = int(re.search(r'\d+', line).group())
                                if n > loadstep:
                                    loadstep = n
                                    overall_progress = 0
                                    self._log.info(line)
                            elif "overall-progress" in line:
                                n = int(re.search(r'\d+', line).group())
                                if n > overall_progress:
                                    overall_progress = n
                                    self._log.info(line)
                        old_tail = new_tail
                time.sleep(update_rate)
        except Exception as e:
            pass

    def exit(self, close_log=True, timeout=3):
        """Exit MAPDL process"""
        if self._exited:
            return

        # self._log.debug('Exiting ANSYS')
        if self._server is not None:
            # attempt to cache final path and lockfile before exiting
            try:
                path = self.directory
                lockfile = self._lockfile
            except:
                pass

            try:
                self.run('/EXIT')
            except:
                pass

            try:
                self._server.terminate()
            except:
                pass
            self._server = None

        if close_log:
            self._close_apdl_log()

        # wait for lockfile to be removed
        if timeout:
            tstart = time.time()
            if lockfile is not None:
                while os.path.isfile(lockfile):
                    time.sleep(0.05)
                    telap = tstart - time.time()
                    if telap > timeout:
                        return 1

        try:
            self._remove_lockfile()
        except:
            pass

        self._exited = True

    def _remove_lockfile(self):
        """Removes lockfile"""
        if os.path.isfile(self._lockfile):
            try:
                os.remove(self._lockfile)
            except:
                pass

    def _run(self, command):
        """Sends a command to the mapdl server via the CORBA interface"""
        self._reset_cache()
        if self._server is None:
            raise MapdlExitedError('ANSYS exited')

        # cleanup command
        command = command.strip()
        if not command:
            raise ValueError('Cannot run empty command')

        if command[:4].lower() == 'cdre':
            with self.non_interactive:
                self.run(command)
            return self._response

        if command[:4].lower() == '/com':
            split_command = command.split(',')
            if len(split_command) < 2:
                return ''
            elif not split_command[1]:
                return ''
            elif split_command[1]:
                if not split_command[1].strip():
                    return ''

        # /OUTPUT not redirected properly in corba
        if command[:4].lower() == '/out':
            items = command.split(',')
            if len(items) > 1:  # redirect to file
                if len(items) > 2:
                    if items[2].strip():
                        filename = '.'.join(items[1:3]).strip()
                    else:
                        filename = '.'.join(items[1:2]).strip()
                else:
                    filename = items[1]

                if filename:
                    if os.path.basename(filename) == filename:
                        filename = os.path.join(self.directory, filename)
                    self._output = filename
                    if len(items) == 5:
                        if items[4].lower().strip() == 'append':
                            self._outfile = open(filename, 'a')
                    else:
                        self._outfile = open(filename, 'w')
                else:
                    self._close_output()

            else:
                self._close_output()
            return ''

        # include error checking
        text = ''
        additional_text = ''

        self._log.debug('Running command %s', command)
        text = self._server.executeCommandToString(command)

        # print suppressed output
        additional_text = self._server.executeCommandToString('/GO')

        # return text, additional_text
        if text == additional_text:
            additional_text = ''

        response = '%s\n%s' % (text, additional_text)

        if self._outfile is not None:
            self._outfile.write(response)

        return response

    def _close_output(self):
        """closes the output file"""
        self._output = ''
        if self._outfile:
            self._outfile.close()
        self._outfile = None
