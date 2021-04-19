"""This module is for threaded implementations of the mapdl interface"""
import threading
import uuid
import shutil
import os
import time
import logging

import numpy as np
from tqdm import tqdm

from ansys.mapdl.core import launch_mapdl
from ansys.mapdl.core.misc import create_temp_dir, threaded_daemon, threaded
from ansys.mapdl.core.launcher import port_in_use, MAPDL_DEFAULT_PORT

LOG = logging.getLogger(__name__)
LOG.setLevel('DEBUG')


def available_ports(n_ports, starting_port=MAPDL_DEFAULT_PORT):
    """Return a list the first ``n_ports`` ports starting from ``starting_port``"""

    port = MAPDL_DEFAULT_PORT
    ports = []
    while port < 65536 and len(ports) < n_ports:
        if not port_in_use(port):
            ports.append(port)
        port += 1

    if len(ports) < n_ports:
        raise RuntimeError('There are not %d available ports between %d and 65536'
                           % (n_ports, starting_port))

    return ports


class LocalMapdlInstance():
    """Local instance of MAPDL to be persistant and restarted at will"""

    def __init__(self, path=None, port=None, **kwargs):
        """Initialize job"""
        self._start_thread = None
        self._spawn_kwargs = kwargs
        self.path = path
        self._locked = False

        self.id = str(uuid.uuid4())
        self._port = port
        self._mapdl = None
        self._spawn_kwargs = kwargs

    def new_path(self):
        """Create a new temporary directory for this instance"""
        self.path = create_temp_dir()
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    @property
    def port(self):
        """Port of the MAPDL instance"""
        return self._port

    @port.setter
    def port(self, port):
        """Set the port of the MAPDL instance"""
        if self.active:
            raise RuntimeError('Cannot set the port of a running instance')
        self._port = port

    @property
    def starting(self):
        """Returns true when job is starting in a background thread."""
        if self._start_thread is not None:
            return self._start_thread.is_alive()
        return False

    def wait_until_started(self):
        """Waits until job has started"""
        if self.starting:
            self._start_thread.join()

    def start(self, wait=True, clean=True, timeout=60):
        """Spawn a mapdl instance"""
        if self.active:
            raise RuntimeError('MAPDL instance is already active')

        # MAPDL does not handle existing paths well
        self._mapdl = None
        if self.path is not None:
            if clean:
                self.clean()
        self.new_path()

        def start_mapdl():
            self._mapdl = launch_mapdl(run_location=self.path,
                                       port=self._port,
                                       start_timeout=timeout,
                                       **self._spawn_kwargs)
            # verify port created is the same as the port specified
            if self._mapdl._port != self.port:
                self._port = self._mapdl._port
                LOG.warning('Different MAPDL port used than assigned')

        if wait:
            start_mapdl()
        else:
            self._start_thread = threading.Thread(target=start_mapdl, daemon=True)
            self._start_thread.start()

    def stop(self, wait=True):
        """Stop this Job

        Parameters
        ----------
        wait : bool, optional
            Wait until the job is canceled.  Default ``True``.

        Returns
        --------
        Thread
            Stop thread when ``wait=False``.

        """
        def _stop():
            self._mapdl.exit()
            self._locked = False

        thread = threading.Thread(target=_stop, daemon=True)
        thread.start()
        if wait:
            thread.join()
        else:
            return thread

    @property
    def mapdl(self):
        """MAPDL instance"""
        return self._mapdl

    @property
    def mapdl_connected(self):
        """MAPDL is connected and alive"""
        if self.mapdl is not None:
            return self.mapdl.is_alive
        return False

    @property
    def active(self):
        """Repeated here as this is used in other classes"""
        return self.mapdl_connected

    def __repr__(self):
        txt = ['MAPDL Local Job']
        # txt.append(f'IP Address       : {self.ip}')
        txt.append(f'Port             : {self.port}')
        txt.append(f'Job ID           : {self.id}')
        txt.append(f'Directory        : {self.path}')
        txt.append(f'MADPL Connected  : {self.mapdl_connected}')
        return '\n'.join(txt)

    def clean(self):
        """Attempt to remove all files from the path"""
        try:
            if os.path.isdir(self.path):
                shutil.rmtree(self.path)
        except:
            pass

    @property
    def locked(self):
        return self._locked

    @locked.setter
    def locked(self, value):
        self._locked = value


class LocalMapdlPool():
    """Create a pool of MAPDL instances.

    Parameters
    ----------
    n_instance : int
        Number of instances to create.

    restart_failed : bool, optional
        Restarts failed instances.  Defaults to ``True``.

    wait : bool, optional
        Wait for pool to be initialized.  Otherwise, pool will start
        in the background and all resources may not be available instantly.

    run_location : str, optional
        Base directory to create additional directories for each MAPDL
        instance.  Defaults to a temporary working directory.

    starting_port : int, optional
        Starting port for the MAPDL instances.  Defaults to 50052.

    progress_bar : bool, optional
        Show a progress bar when starting the pool.  Defaults to
        ``True``.  Will not be shown when ``wait=False``

    restart_failed : bool, optional
        Restarts any failed instances in the pool.

    remove_temp_files : bool, optional
        Removes all temporary files on exit.  Default ``True``.

    **kwargs : Additional Keyword Arguments
        See ``help(ansys.mapdl.launcher.launch_mapdl)`` for a complete
        listing of all additional keyword arguments.

    Examples
    --------
    Simply create a pool of 10 instances to run in the temporary
    directory.

    >>> from ansys.mapdl import LocalMapdlPool
    >>> pool = mapdl.LocalMapdlPool(10)

    Create several instances with 1 CPU each running at the current
    directory within their own isolated directories.

    >>> import os
    >>> my_path = os.getcmd()
    >>> pool = mapdl.LocalMapdlPool(10, nproc=1, run_location=my_path)
    Creating Pool: 100%|████████| 10/10 [00:01<00:00,  1.43it/s]
    """

    def __init__(self, n_instances,
                 wait=True,
                 run_location=None,
                 port=MAPDL_DEFAULT_PORT,
                 progress_bar=True,
                 restart_failed=True,
                 remove_temp_files=True,
                 timeout=120,
                 **kwargs):
        """Initialize several instances of mapdl"""
        self._instances = []
        self._root_dir = run_location
        kwargs['remove_temp_files'] = remove_temp_files
        kwargs['mode'] = 'grpc'
        self._spawn_kwargs = kwargs

        if self._root_dir is not None:
            if not os.path.isdir(self._root_dir):
                os.makedirs(self._root_dir)

        self._instances = []
        self._active = True  # used by pool monitor

        n_instances = int(n_instances)
        if n_instances < 1:
            raise ValueError('Number of instances should be at least one.')

        # grab available ports and specify a temporary directory for
        # each instance
        self._instances = []
        for port in available_ports(n_instances, port):
            run_location = create_temp_dir(self._root_dir)
            job = LocalMapdlInstance(path=run_location, port=port,
                                     **self._spawn_kwargs)
            self._instances.append(job)

        for job in self._instances:
            job.start(wait=False, timeout=timeout)

        if wait:
            if progress_bar:
                pbar = tqdm(total=n_instances, desc='Creating Pool')
            for job in self._instances:
                job.wait_until_started()
                pbar.update(1)
            if progress_bar:
                pbar.close()

        # monitor pool if requested
        if restart_failed:
            self._pool_monitor_thread = self._monitor_pool()

    def map(self, func, iterable=None, progress_bar=True,
            close_when_finished=False, timeout=None, wait=True):
        """Run a function for each instance of mapdl within the pool

        Parameters
        ----------
        func : function
            User function with an instance of ``mapdl`` as the first
            argument.  The remaining arguments should match the number
            of items in each iterable (if any).

        iterable : list, tuple, optional
            An iterable containing a set of arguments for ``func``.
            If None, will run ``func`` once for each instance of
            mapdl.

        progress_bar : bool, optional
            Show a progress bar when running the batch.  Defaults to
            ``True``.

        close_when_finished : bool, optional
            Exit the MAPDL instances when the pool is finished.
            Default ``False``.

        timeout : float, optional
            Maximum runtime in seconds for each iteration.  If
            ``None``, no timeout.  If specified, each iteration will
            be only allowed to run ``timeout`` seconds, and then
            killed and treated as a failure.

        wait : bool, optional
            Block execution until the batch is complete.  Default
            ``True``.

        Returns
        -------
        output : list
            A list containing the return values for ``func``.  Failed
            runs will not return an output.  Since the returns are not
            necessarily in the same order as ``iterable``, you may
            want to add some sort of tracker to the return of your
            user function``func``.

        Examples
        --------
        Run several input files while storing the final routine.  Note
        how the user function to be mapped must use ``mapdl`` as the
        first argument.  The function can have any number of
        additional arguments.

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
        """

        # check if any instances are available
        if not self.n_alive:
            raise RuntimeError('No MAPDL instances available.')

        results = []

        if iterable is not None:
            n = len(iterable)
        else:
            n = len(self)

        pbar = None
        if progress_bar:
            pbar = tqdm(total=n, desc='MAPDL Running')

        @threaded_daemon
        def func_wrapper(inst, func, timeout, args=None):
            """Expect ``inst`` to be an instance

            """
            complete = [False]

            @threaded_daemon
            def run():
                if args is not None:
                    if isinstance(args, (tuple, list)):
                        results.append(func(inst.mapdl, *args))
                    else:
                        results.append(func(inst.mapdl, args))
                else:
                    results.append(func(inst.mapdl))
                complete[0] = True

            run_thread = run()
            if timeout:
                tstart = time.time()
                while not complete[0]:
                    time.sleep(0.01)
                    if (time.time() - tstart) > timeout:
                        break

                if not complete[0]:
                    LOG.error('Killed instance due to timeout of %f seconds', timeout)
                    inst.exit()
            else:
                run_thread.join()
                # finished without completing
                if not complete[0]:
                    try:
                        inst.mapdl.exit()
                    except:
                        pass

            instance.locked = False
            if pbar:
                pbar.update(1)

        threads = []
        if iterable is not None:
            threads = []
            for args in iterable:
                # grab the next available instance of mapdl
                instance = self.next_available()
                instance.locked = True
                threads.append(func_wrapper(instance, func, timeout, args))

            if close_when_finished:
                # start closing any instances that are not in execution
                self.exit()
            else:
                # wait for all threads to complete
                if wait:
                    [thread.join() for thread in threads]

        else:  # simply apply to all
            for instance in self._instances:
                if instance:
                    threads.append(func_wrapper(instance, func, timeout))

            # wait for all threads to complete
            if wait:
                [thread.join() for thread in threads]

        return results

    def run_batch(self, files, clear_at_start=True, progress_bar=True,
                  close_when_finished=False, timeout=None, wait=True):
        """Run a batch of input files on the pool.

        Parameters
        ----------
        files : list
            List of input files to run.

        clear_at_start : bool, optional
            Clear MAPDL at the start of execution.  By default this is
            ``True``, and setting this to ``False`` may lead to
            instability.

        progress_bar : bool, optional
            Show a progress bar when starting the pool.  Defaults to
            ``True``.  Will not be shown when ``wait=False``

        progress_bar : bool, optional
            Show a progress bar when running the batch.  Defaults to
            ``True``.

        close_when_finished : bool, optional
            Exit the MAPDL instances when the pool is finished.
            Default ``False``.

        timeout : float, optional
            Maximum runtime in seconds for each iteration.  If
            ``None``, no timeout.  If specified, each iteration will
            be only allowed to run ``timeout`` seconds, and then
            killed and treated as a failure.

        wait : bool, optional
            Block execution until the batch is complete.  Default
            ``True``.

        Returns
        -------
        outputs : list
            List of text outputs from MAPDL for each batch run.  Not
            necessarily in the order of the inputs. Failed runs will
            not return an output.  Since the returns are not
            necessarily in the same order as ``iterable``, you may
            want to add some sort of tracker or note within the input files.

        Examples
        --------
        Run 20 verification files on the pool

        >>> from ansys.mapdl import examples
        >>> files = [examples.vmfiles['vm%d' % i] for i in range(1, 21)]
        >>> outputs = pool.run_batch(files)
        >>> len(outputs)
        20
        """
        # check all files exist before running
        for filename in files:
            if not os.path.isfile(filename):
                raise FileNotFoundError('Unable to locate file %s' % filename)

        def run_file(mapdl, input_file):
            if clear_at_start:
                mapdl.finish()
                mapdl.clear()
            response = mapdl.input(input_file)
            return response

        return self.map(run_file, files, progress_bar=progress_bar)

    def next_available(self, return_index=False):
        """Wait until an instance of mapdl is available and return that instance.

        Parameters
        --------
        return_index : bool, optional
            Return the index along with the instance.  Default ``False``.

        Returns
        --------
        mapdl : MapdlGrpc
            Instance of MAPDL.

        index : int
            Index within the pool of the instance of MAPDL.  By
            default this is not returned.

        Examples
        --------
        >>> mapdl = pool.next_available()
        >>> print(mapdl)
        Product:         ANSYS Mechanical Enterprise
        MAPDL Version:   RELEASE                    BUILD  0.0      UPDATE        0
        PyANSYS Version: 0.55.1
        """
        # loop until the next instance is available
        while True:
            for i, instance in enumerate(self._instances):
                if not instance.locked and instance.mapdl_connected:
                    # any instance that is not running or exited
                    # should be available
                    if not instance.mapdl.busy:
                        # double check that this instance is alive:
                        try:
                            instance.mapdl.inquire('JOBNAME')
                        except:
                            instance.stop()
                            continue

                        if return_index:
                            return instance, i
                        else:
                            return instance
                    else:
                        instance.mapdl._exited = True

    def __del__(self):
        self.exit()

    def exit(self, block=False):
        """Close out all instances in the pool.

        Parameters
        ----------
        block : bool, optional
            When ``True``, wait until all processes are closed.

        Examples
        --------
        >>> pool.exit()
        """
        self._active = False  # stop any active instance restart
        threads = [inst.stop(wait=False) for inst in self._instances]

        if block:
            [thread.join() for thread in threads]

    @property
    def n_alive(self):
        """Number of instances of MAPDL that are alive"""
        count = 0
        for instance in self._instances:
            if instance.active:
                count += 1
        return count

    def __len__(self):
        return len(self._instances)

    def __getitem__(self, index):
        """Return MAPDL of an instance by an index"""
        return self._instances[index]._mapdl

    def __iter__(self):
        """Iterate through active instances"""
        for instance in self._instances:
            yield instance._mapdl

    @threaded_daemon
    def _monitor_pool(self, refresh=1.0):
        """Checks if instances within a pool have exited (failed) and
        restarts them.
        """
        while self._active:
            for inst in self._instances:
                if not inst.active:
                    try:
                        inst.start(wait=True)
                    except Exception as e:
                        logging.error(e, exc_info=True)
            time.sleep(refresh)

    def __repr__(self):
        return f'Local MAPDL Pool with {len(self)} active instances'

    def stop(self, block=False):
        """Alias for exit"""
        self.exit(block)

    @property
    def _ports(self):
        """Ports used by each instance of MAPDL"""
        ports = []
        for mapdl in self:
            if mapdl is not None:
                ports.append(mapdl._port)
        return ports

    def _verify_unique_ports(self):
        if not len(self) == len(np.unique(self._ports)):
            raise RuntimeError('MAPDL pool ports are non-unique')
