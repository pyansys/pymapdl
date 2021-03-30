"""Contains the MapdlMath classes, allowing for math operations within
mapdl from Python.  """
import os
import weakref
import string
import random
from enum import Enum

import numpy as np
from ansys.grpc.mapdl import ansys_kernel_pb2 as anskernel
from ansys.grpc.mapdl import mapdl_pb2 as pb_types

from .parameters import interp_star_status
from .errors import ANSYSDataTypeError, protect_grpc
from .mapdl_grpc import MapdlGrpc
from .common_grpc import (parse_chunks, ANSYS_VALUE_TYPE,
                          DEFAULT_CHUNKSIZE, DEFAULT_FILE_CHUNK_SIZE)


MYCTYPE = {np.int32: 'I',
           np.int64: 'L',
           np.single: 'S',
           np.double: 'D',
           np.complex64: 'C',
           np.complex128: 'Z'}


NP_VALUE_TYPE = {value: key for key, value in ANSYS_VALUE_TYPE.items()}

# for windows LONG vs INT32
if os.name == 'nt':
    NP_VALUE_TYPE[np.intc] = 1


def id_generator(size=6, chars=string.ascii_uppercase):
    """Generate a random string"""
    return ''.join(random.choice(chars) for _ in range(size))


class ObjType(Enum):
    """Generic APDLMath Object ( Shared features between Vec Mat and
    Solver components """
    GEN = 1
    VEC = 2
    DMAT = 3
    SMAT = 4


def get_nparray_chunks(name, array, chunk_size=DEFAULT_FILE_CHUNK_SIZE):
    """Serializes a numpy array into chunks"""
    stype = NP_VALUE_TYPE[array.dtype.type]
    arr_sz = array.size
    i = 0  # position counter
    byte_array = array.tobytes()
    while i < len(byte_array):
        piece = byte_array[i: i + chunk_size]
        chunk = anskernel.Chunk(payload=piece, size=len(piece))
        yield pb_types.SetVecDataRequest(vname=name, stype=stype, size=arr_sz,
                                         chunk=chunk)
        i += chunk_size

def get_nparray_chunks2(name, array, chunk_size=DEFAULT_FILE_CHUNK_SIZE):
    """Serializes a 2D numpy array into chunks"""
    stype = NP_VALUE_TYPE[array.dtype.type]
    sh1 = array.shape[0]
    sh2 = array.shape[1]
    i = 0  # position counter
    byte_array = array.tobytes(order='F')
    while i < len(byte_array):
        piece = byte_array[i: i + chunk_size]
        chunk = anskernel.Chunk(payload=piece, size=len(piece))
        yield pb_types.SetMatDataRequest( mname=name, stype=stype, nrow=sh1,
                                          ncol=sh2, chunk=chunk)
        i += chunk_size

class MapdlMath():
    """Abstract mapdl math class.  Created from a ``Mapdl`` instance.

    Examples
    --------
    Create an instance from an existing mapdl instance

    >>> import ansys
    >>> mapdl = ansys.Mapdl()
    >>> mm = mapdl.math()

    Alternatively:

    >>> from ansys.mapdl.math import MapdlMath
    >>> mm = MapdlMath(mapdl)  # from mapdl above

    Vector addition

    >>> v1 = mm.ones(10)
    >>> v2 = mm.ones(10)
    >>> v3 = v1 + v2

    Matrix multiplcation

    >>> v1 = mm.ones(10)
    >>> m1 = mm.rand(10, 10)
    >>> v2 = m1*v1

    """

    def __init__(self, mapdl):
        if not isinstance(mapdl, MapdlGrpc):
            raise TypeError('``mapdl`` must be a MapdlGrpc instance')
        self._mapdl_weakref = weakref.ref(mapdl)

    @property
    def _mapdl(self):
        """Return the weakly referenced instance of mapdl"""
        return self._mapdl_weakref()

    def free(self):
        """Delete all vectors"""
        self._mapdl.run("*FREE,ALL")

    def __repr__(self):
        return self._status

    def status(self):
        """Print out the status of all APDLMath Objects"""
        print(self._status)

    @property
    def _status(self):
        """Print out the status of all APDLMath Objects"""
        return self._mapdl.run("*STATUS,MATH")

    def vec(self, size=0, dtype=np.double, init=None, name=None):
        """Create a vector

        Parameters
        ----------
        size : int
            Size of the vector

        dtype : np.dtype, optional
            Datatype of the vector.  Must be either np.int32, np.int64, or np.double

        init : str, optional
            Initialization options.  Can be ones, zeros, or rand.

        Returns
        -------
        vector : ansys.mapdl.math.AnsVec
            ANSYS Vector
        """
        if dtype not in MYCTYPE:
            raise ANSYSDataTypeError

        if not name:
            name = id_generator()
            self._mapdl.run("*VEC," + name + "," + MYCTYPE[dtype] + ",ALLOC," + str(size))
            return AnsVec(name, self._mapdl, dtype, init)
        else:
            return AnsVec(name, self._mapdl)
    
    def mat(self, nrow=0, ncol=0, dtype=np.double, init=None, name=None):
        """Create matrix

        Parameters
        ----------
        nrow : int
            Number of rows.

        ncol : int
            Number of columns.

        dtype : np.dtype, optional
            Datatype of the vector.  Must be either np.int32, np.int64, or np.double

        init : str, optional
            Initialization options.  Can be ones, zeros, or rand.

        Returns
        -------
        matrix : ansys.math.AnsMat
            ANSYS matrix
        """
        if dtype not in MYCTYPE:
            raise ValueError('Invalid datatype.  Must be one of the following:\n' +
                             'np.int32, np.int64, or np.double')

        if not name:
            name = id_generator()
            self._mapdl.run(f'*DMAT,{name},{MYCTYPE[dtype]},ALLOC,{nrow},{ncol}')
            mat = AnsDenseMat(name, self._mapdl)

            if (init == "rand"):
                mat.rand()
            elif (init == "ones"):
                mat.ones()
            elif (init == "zeros"):
                mat.zeros()
            elif init is not None:
                raise ValueError("Invalid init method")
        else:
            info = self._mapdl._data_info(name)
            mtype = info.objtype
            if mtype == 2:
                return AnsDenseMat(name, self._mapdl)
            else:
                return AnsSparseMat(name, self._mapdl)

        return mat

    def zeros(self, nrow, ncol=None, dtype=np.double):
        """Create a vector or matrix containing all zeros

        nrow : int
            Number of rows.

        ncol : int, optional
            Number of columns.  If specified, returns a matrix.

        dtype : np.dtype, optional
            Datatype of the vector.  Must be either np.int32, np.int64, or np.double

        Returns
        -------
        ans_vec_mat : ansys.AnsVec or ansys.AnsMat
            ANSYS vector or matrix depending on if "ncol" is specified

        Examples
        --------
        Create a zero vector

        >>> import ansys
        >>> mapdl = ansys.Mapdl()
        >>> mm = mapdl.math()
        >>> vec = mm.zeros(10)

        Create a zero matrix

        >>> mat = mm.zeros(10, 10)
        """
        if dtype not in MYCTYPE:
            raise ANSYSDataTypeError

        if not ncol:
            return self.vec(nrow, dtype, init="zeros")
        else:
            return self.mat(nrow, ncol, dtype, init="zeros")

    def ones(self, nrow, ncol=None, dtype=np.double):
        """Create a vector or matrix containing all ones

        nrow : int
            Number of rows.

        ncol : int, optional
            Number of columns.  If specified, returns a matrix.

        dtype : np.dtype, optional
            Datatype of the vector.  Must be either np.int32, np.int64, or np.double

        Returns
        -------
        ans_vec_mat : ansys.AnsVec or ansys.AnsMat
            ANSYS vector or matrix depending on if "ncol" is specified

        Examples
        --------
        Create a ones vector

        >>> import ansys
        >>> mapdl = ansys.Mapdl()
        >>> mm = mapdl.math()
        >>> vec = mm.ones(10)

        Create a ones matrix

        >>> mat = mm.ones(10, 10)
        """

        if not ncol:
            return self.vec(nrow, dtype, init="ones")
        else:
            return self.mat(nrow, ncol, dtype, init="ones")

    def rand(self, nrow, ncol=None, dtype=np.double):
        """Create a vector or matrix containing all random values

        nrow : int
            Number of rows.

        ncol : int, optional
            Number of columns.  If specified, returns a matrix.

        dtype : np.dtype, optional
            Datatype of the vector.  Must be either np.int32, np.int64, or np.double

        Returns
        -------
        ans_vec_mat : ansys.AnsVec or ansys.AnsMat
            ANSYS vector or matrix depending on if "ncol" is specified

        Examples
        --------
        Create a random vector

        >>> import ansys
        >>> mapdl = ansys.Mapdl()
        >>> mm = mapdl.math()
        >>> vec = mm.rand(10)

        Create a random matrix

        >>> mat = mm.rand(10, 10)
        """
        if not ncol:
            return self.vec(nrow, dtype, init="rand")
        else:
            return self.mat(nrow, ncol, dtype, init="rand")

    def matrix(self, matrix, name=None, triu=False):
        """Send a scipy matrix or numpy array to MAPDL.

        Parameters
        ----------
        matrix : np.ndarray
            Numpy array to send as a matrix to MAPDL.

        name : str, optional
            APDLMath matrix name.  Optional and defaults to a random name.

        triu: bool, optional
            ``True`` when the matrix is upper triangular, ``False``
            when unsymmetric.

        Returns
        -------
        ans_mat : MapdlMath.AnsMat
            MapdlMath matrix.

        Examples
        --------
        Generate a random sparse matrix

        >>> from scipy import sparse
        >>> sz = 5000
        >>> mat = sparse.random(sz, sz, density=0.05, format='csr')
        >>> ans_mat = mm.matrix(mat, name)
        >>> ans_mat
        APDLMath Matrix 5000 x 5000

        Read the matrix back to Python

        >>> ans_mat.asarray()
        <500x5000 sparse matrix of type '<class 'numpy.float64'>'
                with 1250000 stored elements in Compressed Sparse Row format>

        """
        if name is None:
            name = id_generator()
        elif not isinstance(name, str):
            raise TypeError('``name`` parameter must be a string')
        
        from scipy import sparse
        self._set_mat(name, matrix, triu)
        if ( sparse.issparse(matrix)) :
            return AnsSparseMat(name, self._mapdl)
        else:
            return AnsDenseMat(name, self._mapdl)

    def load_matrix_from_file(self, type=np.double, fname="file.full", matId="STIFF"):
        """Load a matrix from a file"""
        name = id_generator()
        self._mapdl._log.info(f"Calling MAPDL to extract the %s matrix from %s",
                              matId, fname)
        self._mapdl.run(f"*SMAT,{name},{MYCTYPE[type]},IMPORT,FULL,{fname},{matId}")
        return AnsSparseMat(name, self._mapdl)

    def stiff(self, dtype=np.double, fname="file.full"):
        """Load the stiffness matrix from the full file

        Examples
        --------
        >>> k = mapdl.math.stiff()
        APDLMATH Matrix 60 x 60

        Convert to a scipy array

        >>> mat = k.asarray()
        >>> mat
        <60x60 sparse matrix of type '<class 'numpy.float64'>'
            with 1734 stored elements in Compressed Sparse Row format>
        """
        return self.load_matrix_from_file(dtype, fname, "STIFF")

    def mass(self, dtype=np.double, fname="file.full"):
        """Load the mass matrix from the full file

        Examples
        --------
        >>> m = mapdl.math.mass()
        >>> m
        APDLMATH Matrix 60 x 60

        Convert to a scipy array

        >>> mat = m.asarray()
        >>> mat
        <60x60 sparse matrix of type '<class 'numpy.float64'>'
            with 1734 stored elements in Compressed Sparse Row format>
        """
        return self.load_matrix_from_file(dtype, fname, "MASS")

    def damp(self, dtype=np.double, fname="file.full"):
        """Load the damping matrix from the full file"""
        return self.load_matrix_from_file(dtype, fname, "DAMP")

    def getVec(self, dtype=np.double, fname="file.full", matId="RHS"):
        """Load a vector from a file"""
        name = id_generator()
        self._mapdl._log.info("Call MAPDL to extract the %s vector from the file %s",
                              matId, fname)
        self._mapdl.run(f"*VEC,{name},{MYCTYPE[dtype]},IMPORT,FULL,{fname},{matId}")
        return AnsVec(name, self._mapdl)

    def set_vec(self, vname, data):
        """Push a numpy array or Python list to the MAPDL Memory
        Workspace.

        Parameters
        ----------
        vname : str
            APDLMath vector name

        data : np.ndarray, list
            Numpy array or Python list to push to MAPDL.  Must be 1
            dimensional.

        Returns
        -------
        vec : ansys.mapdl.math.AnsVec
            MAPDL vector instance generated from the pushed vector.

        Examples
        --------
        >>> data = np.random.random(10)
        >>> vec = mm.set_vec('NewVec', data)
        >>> np.isclose(vec.asarray(), data)
        True
        """
        self._set_vec(vname, data)
        return AnsVec(vname, self._mapdl)

    def rhs(self, type=np.double, fname="file.full"):
        return self.getVec(type, fname, "RHS")

    def eigs(self, nev, k, m=None, c=None, phi=None, algo=None,
             fmin=None, fmax=None):
        """Solve an eigenproblem

        Parameters
        ----------
        nev : int
            Number of eigenvalues to compute.

        K : ansys.AnsMat
            An array representing the operation ``A * x`` where A is a
            square matrix.

        M : ansys.AnsMat, optional
            An array representing the operation ``M * x`` for the
            generalized eigenvalue problem:

            ``K * x = M * x``

        Examples
        --------
        Solve an eigenproblem using the mass and stiffness matrices
        stored from a prior ansys run.

        >>> k = mm.stiff()
        >>> m = mm.mass()
        >>> nev = 10
        >>> a = mm.mat(k.nrow, nev)
        >>> ev = mm.eigs(nev, k, m, phi=a)
        """
        if not fmin:
            fmin = ''
        if not fmax:
            fmax = ''

        cid = ''

        if not c:
            if (k.sym() and m.sym()):
                if not algo:
                    algo = "LANB"
            else:
                algo = "UNSYM"
        else:
            cid = c.id
            algo = "DAMP"

        self._mapdl.run("/SOLU")
        self._mapdl.run("antype,modal")
        self._mapdl.run("modopt," + algo + "," + str(nev) + "," + str(fmin) + "," + str(fmax))
        ev = self.vec()

        phistr = '' if not phi else phi.id
        self._mapdl.run("*EIG," + k.id + "," + m.id + "," + cid + "," + ev.id + "," + phistr)

        return ev

    def dot(self, vec_a, vec_b):
        """Dot product between two ANSYS vector objects.

        Parameters
        ----------
        vec_a : ansys.mapdl.math.AnsVec
            Ansys vector object.

        vec_b : ansys.mapdl.math.AnsVec
            Ansys vector object.

        Returns
        -------
        dot_product : float
            Dot product between the two vectors.

        Examples
        --------
        >>> v = mm.ones(10)
        >>> w = mm.ones(10)
        >>> v.dot(w)
        """
        return dot(vec_a, vec_b)

    def add(self, obj1, obj2):
        """Add two APDLMath vectors or matrices.

        Parameters
        ----------
        obj1 : ansys.mapdl.math.AnsVec or ansys.mapdl.math.AnsMat
            Ansys object.

        obj2 : ansys.mapdl.math.AnsVec or ansys.mapdl.math.AnsMat
            Ansys object.

        Returns
        -------
        objout : ansys.mapdl.math.AnsVec or ansys.mapdl.math.AnsMat
            Sum of the two input objects.  Type of the output matches type of the input.
            Sum of the two vectors/matrices.

        Examples
        --------
        >>> v = mm.ones(10)
        >>> w = mm.ones(10)
        >>> x = mm.add( v, w)
        """
        return obj1 + obj2

    def subtract(self, obj1, obj2):
        """Subtract two ANSYS vectors or matrices.

        Parameters
        ----------
        obj1 : ansys.mapdl.math.AnsVec or ansys.mapdl.math.AnsMat
            Ansys object.

        obj2 : ansys.mapdl.math.AnsVec or ansys.mapdl.math.AnsMat
            Ansys object.

        Returns
        -------
        objout : ansys.mapdl.math.AnsVec or ansys.mapdl.math.AnsMat
            Difference of the two input vectors or matrices.  Type of the output
            matches the type of the input.

        Examples
        --------
        >>> v = mm.ones(10)
        >>> w = mm.ones(10)
        >>> x = mm.add( v, w)
        """
        return obj1 - obj2

    
    def factorize(self, mat):
        """Factorize a matrix

        Parameters
        ----------
        mat : ansys.mapdl.math.AnsMat
            A ansys.mapdl.math matrix

        Examples
        --------
        >>> mm = mapdl.math
        >>> dim = 1000
        >>> m2 = mm.rand(dim, dim)
        >>> m3 = m2.copy()
        >>> mat = mm.factorize(m2)
        
        """
        solver = AnsSolver(id_generator(), self._mapdl)
        solver.factorize(mat)
        return solver

    def norm(self, obj, ord="nrm2"):
        """ Matrix or vector norm

        Parameters
        ----------
        obj : ansys.mapdl.math.AnsMat or ansys.mapdl.math.AnsVec
        ord : Order of the norm. nrm2(default), nrminf, nrm1

        Examples
        --------
        >>> mm = mapdl.math
        >>> dim = 1000
        >>> m2 = mm.rand(dim, dim)
        >>> nrm = mm.norm( m2)
        """
        obj.norm()

    @protect_grpc
    def _set_vec(self, vname, arr, dtype=None, chunk_size=DEFAULT_CHUNKSIZE):
        """Transfer a numpy array to MAPDL as a MAPDL Math vector.

        Parameters
        ----------
        vname : str
            Vector parameter name.  Character ":" is not allowed.

        arr : np.ndarray
            Numpy array to upload

        dtype : np.dtype, optional
            Type to upload array as.  Defaults to the current array type.

        chunk_size : int, optional
            Chunk size in bytes.  Must be less than 4MB.

        """
        if ':' in vname:
            raise ValueError('The character ":" is not permitted in a MAPDL MATH'
                             ' vector parameter name')
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        if dtype is not None:
            if arr.dtype != dtype:
                arr = arr.astype(dtype)

        if arr.dtype not in list(NP_VALUE_TYPE.keys()):
            dtypes = list(NP_VALUE_TYPE.keys())
            if None in dtypes:
                dtypes.remove(None)
            raise TypeError('Invalid array datatype.  Must be one of the following:\n'
                            + ('\n'.join([str(dtype) for dtype in dtypes])))
        chunks_generator = get_nparray_chunks(vname, arr, chunk_size)
        self._mapdl._stub.SetVecData(chunks_generator)

    @protect_grpc
    def _set_mat(self, mname, arr, sym, dtype=None, chunk_size=DEFAULT_CHUNKSIZE):
        """Transfer a 2D dense of scipy array to MAPDL as a MAPDL Math matrix

        Parameters
        ----------
        mname : str
            Matrix parameter name.  Character ":" is not allowed.

        arr : 
            Matrix to upload

        sym : bool
            ``True`` when matrix is symmetric. Unused if Matrix is dense

        dtype : np.dtype, optional
            Type to upload array as.  Defaults to the current array type.

        chunk_size : int, optional
            Chunk size in bytes.  Must be less than 4MB.

        """
        from scipy import sparse

        if ':' in mname:
            raise ValueError('The character ":" is not permitted in a MAPDL MATH'
                             ' matrix parameter name')
        if not len(mname):
            raise ValueError('Empty MAPDL matrix name not permitted')

        if isinstance(arr, np.ndarray):
            if arr.ndim == 1:
                raise ValueError('Input appears to be an array.  '
                                 'Use ``set_vec`` instead.)')
            if arr.ndim > 2:
                raise ValueError('Dense arrays must be 2 dimensional')

        if sparse.issparse(arr):
            arr = sparse.csr_matrix(arr)

            if arr.shape[0] != arr.shape[1]:
                raise ValueError('APDLMath only supports square matrices')

                if dtype is not None:
                    if arr.data.dtype != dtype:
                        arr.data = arr.data.astype(dtype)

                if arr.data.dtype not in list(NP_VALUE_TYPE.keys()):
                    dtypes = list(NP_VALUE_TYPE.keys())
                    if None in dtypes:
                        dtypes.remove(None)
                        raise TypeError('Invalid array datatype.  Must be one of the following:\n'
                                        + ('\n'.join([str(dtype) for dtype in dtypes])))

            # The data vector
            dataname = f'{mname}_DATA'
            self.set_vec(dataname, arr.data)

            # indptr vector
            indptrname = f'{mname}_IND'
            indv = arr.indptr.astype('int64') + 1
            self.set_vec(indptrname, indv)

            # indices vector
            indxname = f'{mname}_PTR'
            idx = arr.indices + 1
            self.set_vec(indxname, idx)

            flagsym = 'FALSE'
            if sym is True:
                flagsym = 'TRUE'
                self._mapdl.run(f'*SMAT,{mname},D,ALLOC,CSR,{indptrname},{indxname},{dataname},{flagsym}')
        else:

            if self._mapdl._grpc_api_ver < 0.4:
                    raise TypeError('This feature is not supported.\n')                
                
            if dtype is not None:
                if arr.dtype != dtype:
                    arr = arr.astype(dtype)

            if arr.dtype not in list(NP_VALUE_TYPE.keys()):
                dtypes = list(NP_VALUE_TYPE.keys())
                if None in dtypes:
                    dtypes.remove(None)
                    raise TypeError('Invalid array datatype.  Must be one of the following:\n'
                                    + ('\n'.join([str(dtype) for dtype in dtypes])))

            chunks_generator = get_nparray_chunks2(mname, arr, chunk_size)
            self._mapdl._stub.SetMatData(chunks_generator)
            
            

class ApdlMathObj:
    def __init__(self, id, mapdl, type=ObjType.GEN):
        self.id = id
        self._mapdl = mapdl
        self.type = type

    #def __del__(self):
    #    self._mapdl._log.debug("Deleting the MAPDL Vector Object")
    #    self._mapdl.run("*FREE," + self.id)

    def __repr__(self):
        return "APDLMath Object %s" % str(self.id)

    def __str__(self):
        return self._mapdl.run("*PRINT," + self.id)

    def copy(self):
        """Returns the name of the copy of this object"""
        name = id_generator()  # internal name of the new vector
        if self.type == ObjType.VEC:
            acmd = "*VEC"
        elif self.type == ObjType.DMAT:
            acmd = "*DMAT"
        elif self.type == ObjType.SMAT:
            acmd = "*SMAT"
        else:
            raise TypeError("Unknown obj type: operation aborted")

        # APDLMath cmd to COPY vin to vout
        self._mapdl.run(acmd + "," + name + ",D,COPY," + self.id)
        return name

    def _init(self, method):
        cmd = "*INIT," + self.id + "," + method
        self._mapdl.run(cmd)

    def zeros(self):
        """Set all values of the vector to zero"""
        return self._init("ZERO")

    def ones(self):
        """Set all values of the vector to one"""
        return self._init("CONST,1")

    def rand(self):
        """Set all values of the vector to a random number"""
        return self._init("RAND")

    def const(self, value):
        """Set all values of the vector to a constant"""
        return self._init("CONST," + str(value))

    def norm(self, nrmtype="nrm2"):
        """Matrix or vector norm.

        Parameters
        ----------
        nrmtype : str, optional
            Mathematical norm to use.  One of:

            - ``'NRM2'``: L2 (Euclidean or SRSS) norm (default).
            - ``'NRM1'``: L1 (absolute sum) norm (vectors only).
            - ``'NRMINF'`` Maximum norm.

        Returns
        -------
        nrm : float
            Norm of the matrix or vector(s).

        Examples
        --------
        >>> mm = mapdl.math
        >>> dim = 1000
        >>> m2 = mm.rand(dim, dim)
        >>> nrm = mm.norm( m2)
        """
        val_name = 'py_val'
        self._mapdl.run("*NRM," + self.id + "," + nrmtype + ",%s" % val_name)
        return self._mapdl.scalar_param(val_name)

    def axpy(self, op, val1, val2):
        """Perform the matrix operation: ``M2= v*M1 + w*M2``"""
        cmd = "*AXPY," + str(val1) + ",0," + op.id + "," + str(val2) + ",0," + self.id
        self._mapdl._log.info(f">> Call Mapdl to perform AXPY operation")
        self._mapdl.run(cmd)
        return self

    def __add__(self, op2):
        opout = self.copy()
        self._mapdl._log.info(">> Call Mapdl to perform AXPY operation")
        self._mapdl.run(f"*AXPY,1,0,{op2.id},1,0,{opout.id}")
        return opout

    def __sub__(self, op2):
        opout = self.copy()
        self._mapdl._log.info("Call Mapdl to perform AXPY operation")
        self._mapdl.run(f"*AXPY,-1,0,{op2.id},1,0,{opout.id}")
        return opout

    def __iadd__(self, op):
        return self.axpy(op, 1, 1)

    def __isub__(self, op):
        return self.axpy(op, -1, 1)

    def __imul__(self, val):
        self._mapdl._log.info("Call Mapdl to scale the object")
        self._mapdl.run(f"*SCAL,{self.id},{val}")
        return self

    def __itruediv__(self, val):
        if val == 0:
            raise ZeroDivisionError('division by zero')
        self._mapdl._log.info("Call Mapdl to 1/scale the object")
        self._mapdl.run(f"*SCAL,{self.id},{1/val}")
        return self

    @property
    @protect_grpc
    def _data_info(self):
        """Return the data type of a parameter"""
        request = pb_types.ParameterRequest(name=self.id)
        return self._stub.GetDataInfo(request)


class AnsVec(ApdlMathObj):
    """APDLMath Vector Object"""
    def __init__(self, id, mapdl, dtype=np.double, init=None):
        ApdlMathObj.__init__(self, id, mapdl, ObjType.VEC)

        if init not in ['ones', 'zeros', 'rand', None]:
            raise ValueError('Invalid init option.  Should be "ones", "zeros", "rand", or None')

        if (init == "rand"):
            self.rand()
        elif (init == "ones"):
            self.ones()
        elif (init == "zeros"):
            self.zeros()

    @property
    def size(self):
        return int(self._mapdl.scalar_param(self.id + "_DIM"))

    def __repr__(self):
        return "APDLMath Vector Size %s" % self.size

    def __getitem__(self, num):
        if num < 0:
            raise ValueError('Negative indices not permitted')
        cmd = "pyval=" + self.id + "(" + str(num+1) + ")"
        self._mapdl.run(cmd)
        return self._mapdl.scalar_param("pyval")

    def __mul__(self, vec):
        """Element-Wise product with another ansys vector object
           ( also knowm as Hadamard product )
        Parameters
        ----------
        vec : ansys.mapdl.math.AnsVec
            Ansys vector object.

        Returns
        -------
        hadamard_product : Ansys vector object
            Hadamard product between this vector and the other vector.
        """

        if self._mapdl._grpc_api_ver < 0.4:        
            raise AttributeError('Array multiplication is not yet available.  '
                                 'For dot product, please use `dot()`')

        if not isinstance(vec, AnsVec):
            raise TypeError('Must be an Ansys vector object')
            
        name = id_generator()  # internal name of the new vector/matrix
        info = self._mapdl._data_info(self.id)
        dtype = ANSYS_VALUE_TYPE[info.stype]

        # check size consistency
        if self.size != vec.size :
            raise TypeError('Ansys vectors have inconsistent sizes')

        self._mapdl.run(f"*VEC,{name},{MYCTYPE[dtype]},ALLOC,{info.size1}")
        objout = AnsVec(name, self._mapdl)

        cmd = "*HPROD," + self.id + "," + vec.id + "," + name
        self._mapdl.run(cmd)
        return objout
        
    def copy(self):
        """Return a copy of this vector"""
        return AnsVec(ApdlMathObj.copy(self), self._mapdl)

    def dot(self, vec):
        """Dot product with another ansys vector object

        Parameters
        ----------
        vec : ansys.mapdl.math.AnsVec
            Ansys vector object.

        Returns
        -------
        dot_product : float
            Dot product between this vector and the other vector.
        """
        if not isinstance(vec, AnsVec):
            raise TypeError('Must be an Ansys vector object')

        cmd = "*DOT," + self.id + "," + vec.id + ",py_val"
        self._mapdl.run(cmd)
        return self._mapdl.scalar_param("py_val")

    def asarray(self):
        """Returns vector as a numpy array

        Examples
        --------
        >>> from ansys.mapdl import Mapdl
        >>> mm = mapdl.math()
        >>> v = mm.ones(10)
        >>> v.asarray()
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        """
        return self._mapdl._vec_data(self.id)

    def __array__(self):
        """Allow numpy to access this object as if it was an array"""
        return self.asarray()


class AnsMat(ApdlMathObj):
    """APDLMath Matrix Object"""

    def __init__(self, id, mapdl, type=ObjType.DMAT):
        ApdlMathObj.__init__(self, id, mapdl, type)

    @property
    def nrow(self):
        return int(self._mapdl.scalar_param(self.id + "_ROWDIM"))

    @property
    def ncol(self):
        return int(self._mapdl.scalar_param(self.id + "_COLDIM"))

    @property
    def size(self):
        return self.nrow*self.ncol

    @property
    def shape(self):
        """Returns a numpy-like shape.

        Tuple of (rows and columns)
        """
        return (self.nrow, self.ncol)

    def sym(self):  # BUG this is not always true
        return True

    def asarray(self):
        """Returns vector as a numpy array

        Examples
        --------
        >>> from ansys.mapdl import Mapdl
        >>> mm = mapdl.math()
        >>> v = mm.ones(10)
        >>> v.asarray()
        [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
        """
        return self._mapdl._mat_data(self.id)

    def __repr__(self):
        return "APDLMath Matrix %d x %d" % (self.nrow, self.ncol)

    def __mul__(self, vec):
        raise AttributeError('Array multiplication is not yet available.  '
                             'For dot product, please use `dot()`')

    def dot(self, obj):
        """Perform the matrix multiplication by another vector or matrix

        Parameters
        ----------
        obj : ansys.mapdl.math.AnsVec or ansys.mapdl.math.AnsMat
            Ansys object.

        Returns
        -------
        dot_product : ansys.mapdl.math.AnsVec or ansys.mapdl.math.AnsMat
            Matrix multiplication result.

        Examples
        --------
        Multiplication of a matrix and vector

        >>> m1 = mm.rand(10, 10)
        >>> v1 = mm.rand(10)
        >>> v2 = m1.dot(v1)
        >>> assert np.allclose(m1.asarray() @ v1.asarray(), v2)

        """
        name = id_generator()  # internal name of the new vector/matrix
        info = self._mapdl._data_info(self.id)
        dtype = ANSYS_VALUE_TYPE[info.stype]
        if obj.type == ObjType.VEC:
            self._mapdl.run(f"*VEC,{name},{MYCTYPE[dtype]},ALLOC,{info.size1}")
            objout = AnsVec(name, self._mapdl)
        else:
            self._mapdl.run(f"*DMAT,{name},{MYCTYPE[dtype]},ALLOC,{info.size1},{info.size2}")
            objout = AnsDenseMat(name, self._mapdl)
            
        cmd = f"*MULT,{self.id},,{obj.id},,{name}"
        self._mapdl._log.info("Call Mapdl to perform MV Product")
        self._mapdl.run(cmd)
        return objout

    def __getitem__(self, num):
        name = id_generator()
        cmd = "*VEC," + name + ",D,LINK," + self.id + "," + str(num+1)
        self._mapdl.run(cmd)
        return AnsVec(name, self._mapdl)

    @property
    def T(self):
        """Returns the transpose of a MAPDL matrix.

        Examples
        --------
        >>> from ansys.mapdl import Mapdl
        >>> mm = mapdl.math()
        >>> mat = mm.rand(2, 3)
        >>> mat_t = mat.T

        """
        info = self._mapdl._data_info(self.id)

        if info.objtype == 2:
            objtype = "*DMAT"
        else:
            objtype = "*SMAT"

        dtype = ANSYS_VALUE_TYPE[info.stype]
        name = id_generator()
        self._mapdl._log.info("Call MAPDL to transpose")
        self._mapdl.run(f"{objtype},{name},{MYCTYPE[dtype]},COPY,{self.id},TRANS")
        if info.objtype == 2:
            mat = AnsDenseMat(name, self._mapdl)
        else:
            mat = AnsSparseMat(name, self._mapdl)
        return mat


class AnsDenseMat(AnsMat):
    """Dense APDLMath Matrix"""

    def __init__(self, uid, mapdl):
        AnsMat.__init__(self, uid, mapdl, ObjType.DMAT)

    def __array__(self):
        """Allow numpy to access this object as if it was an array"""
        return self.asarray()

    def __repr__(self):
        return f'Dense APDLMath Matrix ({self.nrow}, {self.ncol})'

    def copy(self):
        """Return a copy of this matrix"""
        return AnsDenseMat(ApdlMathObj.copy(self), self._mapdl)


class AnsSparseMat(AnsMat):
    """Sparse APDLMath Matrix"""

    def __init__(self, uid, mapdl):
        AnsMat.__init__(self, uid, mapdl, ObjType.SMAT)

    def __repr__(self):
        return f'Sparse APDLMath Matrix ({self.nrow}, {self.ncol})'

    def copy(self):
        """Return a copy of this matrix"""
        return AnsSparseMat(ApdlMathObj.copy(self), self._mapdl)

    def todense(self):
        """Return this array as a dense np.ndarray"""
        return self.asarray().todense()

    def __array__(self):
        """Allow numpy to access this object as if it was an array"""
        return self.todense()


class AnsSolver(ApdlMathObj):
    """APDLMath Solver Class"""
    def __repr__(self):
        return "APDLMath Linear Solver"

    # def __del__(self):
    #     self._mapdl._log.debug("Deleting the MAPDL Solver Object")
    #     self._mapdl.run(f"*FREE,{self.id}")

    def factorize(self, M, Algo=None):
        """Factorize a matrix

        Performs the numerical factorization of a linear solver system.

        Parameters
        ----------
        """
        if not Algo:
            if (M.type == ObjType.DMAT):
                Algo = "LAPACK"
            elif (M.type == ObjType.SMAT):
                Algo = "DSP"
        self._mapdl.run(f"*LSENGINE,{Algo},{self.id},{M.id}")
        self._mapdl._log.info("Factorizing using the %s package", Algo)
        self._mapdl.run("*LSFACTOR," + self.id)

    def solve(self, B, X=None):
        if not X:
            X = B.copy()
        self._mapdl._log.info("Solving")
        self._mapdl.run("*LSBAC," + self.id + "," + B.id + "," + X.id)
        return X


def rand(obj):
    """Set all values of a mapdl math object to random values

    Parameters
    ----------
    obj : ansys.math.MapdlMath object
        MapdlMath object
    """
    obj._mapdl.cmd("*INIT," + obj.id + ",RAND")


def solve(Mat, B, X=None, Algo=None):
    Solver = AnsSolver(id_generator(), Mat._mapdl)
    Solver.factorize(Mat, Algo)
    if not X:
        X = B.copy()
    X = Solver.solve(B, X)

    del Solver
    return X


def dot(vec1, vec2):
    """Dot product between two ANSYS vector objects

    Parameters
    vec1 : ansys.mapdl.math.AnsVec
        Ansys vector object.

    vec1 : ansys.mapdl.math.AnsVec
        Ansys vector object.

    Returns
    -------
    dot_product : float
        Dot product between the two vectors

    """
    if (vec1.type != ObjType.VEC or vec2.type != ObjType.VEC):
        raise Exception("Both objects must be ANSYS vectors")

    mapdl = vec1._mapdl
    mapdl.run("*DOT," + vec1.id + "," + vec2.id + ",py_val")
    return mapdl.scalar_param("py_val")


def interp_star_status(status, ignore_hidden=True):
    """Interprets STATUS command output from MAPDL

    Parameters
    ----------
    status : str
        Output from APDL *STATUS

    Returns
    -------
    parameters : dict
        Dictionary of parameters.
    """
    parameters = {}
    header = 'Name                   Type            '
    for line in status.splitlines():
        if header in line:
            continue

        try:
            name, mtype, size_mbytes, dim, workspace = line.split()
        except:
            continue

        if ignore_hidden:
            if name[-5:] == '_DATA':
                continue
            elif name[-4:] == '_IND':
                continue
            elif name[-4:] == '_PTR':
                continue

        parameters[name] = {'type': mtype,
                            'size mb': size_mbytes,
                            'dim': dim,
                            'workspace': workspace}

    return parameters
