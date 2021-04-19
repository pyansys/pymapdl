from pathlib import Path
import os
import time

import numpy as np
import pytest

from ansys.mapdl.core.pool import LocalMapdlInstance
from ansys.mapdl.core.misc import get_ansys_bin
from ansys.mapdl.core import LocalMapdlPool, examples
from ansys.mapdl.core.launcher import get_start_instance

# check for a valid MAPDL install with gRPC
valid_rver = ['211']
EXEC_FILE = None
for rver in valid_rver:
    if os.path.isfile(get_ansys_bin(rver)):
        EXEC_FILE = get_ansys_bin(rver)
        break

IGNORE_POOL = os.environ.get('IGNORE_POOL', '').upper() == 'TRUE'

skip_launch_mapdl = pytest.mark.skipif(get_start_instance() is False or IGNORE_POOL,
                                       reason="Must be able to launch MAPDL locally")

TWAIT = 90


@pytest.fixture(scope="module")
def pool():
    mapdl_pool = LocalMapdlPool(4, exec_file=EXEC_FILE, nproc=1)
    yield mapdl_pool

    ##########################################################################
    # test exit
    mapdl_pool.exit()

    timeout = time.time() + TWAIT

    while len(mapdl_pool) != 0:
        time.sleep(0.1)
        if time.time() > timeout:
            raise TimeoutError(f'Failed to restart instance in {TWAIT} seconds')

    assert len(mapdl_pool) == 0

    # check it's been cleaned up
    if mapdl_pool[0] is not None:
        pth = mapdl_pool[0].directory
        if mapdl_pool._spawn_kwargs['remove_temp_files']:
            assert not list(Path(pth).rglob("*.page*"))


@skip_launch_mapdl
def test_instance():
    inst = LocalMapdlInstance()
    inst.start()
    assert inst.active
    assert inst.mapdl_connected
    assert os.path.isdir(inst.path)

    # test basic commands
    inst.mapdl.finish(mute=True)
    inst.mapdl.prep7(mute=True)
    assert inst.mapdl.parameters.routine == 'PREP7'

    # test stop
    inst.stop()
    assert not inst.mapdl_connected
    # inst.clean()
    # assert not os.path.isdir(inst.path)

    # test restart
    inst.start()
    inst.mapdl.finish(mute=True)
    inst.mapdl.prep7(mute=True)
    assert inst.mapdl.parameters.routine == 'PREP7'

    inst.stop()


@skip_launch_mapdl
def test_heal(pool):
    pool_sz = len(pool)
    pool[0].exit()
    pool[1].exit()
    pool[2].exit()

    time.sleep(1)  # wait for shutdown
    timeout = time.time() + TWAIT
    while len(pool) < pool_sz:
        time.sleep(0.1)
        if time.time() > timeout:
            raise TimeoutError(f'Failed to restart instance in {TWAIT} seconds')

    assert len(pool) == pool_sz
    pool._verify_unique_ports()


@skip_launch_mapdl
def test_simple_map(pool):
    pool_sz = len(pool)
    _ = pool.map(lambda mapdl: mapdl.prep7())
    assert len(pool) == pool_sz


@skip_launch_mapdl
def test_map_timeout(pool):
    pool_sz = len(pool)

    def func(mapdl, tsleep):
        mapdl.clear()
        mapdl.prep7()
        time.sleep(tsleep)
        mapdl.post1()
        return tsleep

    timeout = 2
    times = np.array([0, 1, 3, 4])
    output = pool.map(func, times, timeout=timeout)
    assert len(output) == (times < timeout).sum()

    # wait for the pool to heal before continuing
    timeout = time.time() + TWAIT
    while len(pool) < pool_sz:
        time.sleep(0.1)
        if time.time() > timeout:
            raise TimeoutError(f'Failed to restart instance in {TWAIT} seconds')

    assert len(pool) == pool_sz


@skip_launch_mapdl
def test_simple(pool):
    pool_sz = len(pool)

    def func(mapdl):
        mapdl.clear()

    outs = pool.map(func)
    assert len(outs) == len(pool)
    assert len(pool) == pool_sz


# fails intermittently
@skip_launch_mapdl
def test_batch(pool):
    input_files = [examples.vmfiles['vm%d' % i] for i in range(1, 11)]
    outputs = pool.run_batch(input_files)
    assert len(outputs) == len(input_files)


# fails intermittently
@skip_launch_mapdl
def test_map(pool):
    completed_indices = []
    def func(mapdl, input_file, index):
        # input_file, index = args
        print(len(pool))
        mapdl.clear()
        output = mapdl.input(input_file)
        completed_indices.append(index)
        return mapdl.parameters.routine
    inputs = [(examples.vmfiles['vm%d' % i], i) for i in range(1, 11)]
    outputs = pool.map(func, inputs, progress_bar=True, wait=True)

    assert len(outputs) == len(inputs)


@skip_launch_mapdl
def test_abort(pool, tmpdir):
    pool_sz = len(pool)  # initial pool size

    old_paths = [mapdl.directory for mapdl in pool]

    tmp_file = str(tmpdir.join('woa.inp'))
    with open(tmp_file, 'w') as f:
        f.write("EXIT")

    input_files = [examples.vmfiles['vm%d' % i] for i in range(1, 11)]
    input_files += [tmp_file]

    outputs = pool.run_batch(input_files)
    assert len(outputs) == len(input_files) - 1

    # ensure failed instance restarts
    timeout = time.time() + TWAIT
    while len(pool) < pool_sz:
        time.sleep(0.1)
        if time.time() > timeout:
            raise TimeoutError(f'Failed to restart instance in {TWAIT} seconds')

    assert len(pool) == pool_sz

    # verify the temporary directory has been cleaned up for one of the instances
    for path in old_paths:
        path_deleted = os.path.isdir(path)
        if path_deleted:
            break

    assert path_deleted
