
from ctypes import *
from ctypes.util import find_library

import numpy as np
import scipy as sp
from scipy import sparse

from SofaPython import SofaNumpy as sofa_numpy

from contextlib import contextmanager

def set_opaque(obj, name, value):
    '''set an opaque data to ctypes value'''
    value_type = type(value)

    data = obj.findData(name)
    ptr, shape, typename = data.getValueVoidPtr()

    class Opaque(Structure):
        _fields_ = (('data', value_type), )

    dst = Opaque.from_address(ptr)
    dst.data = value


def Vec(t):
    '''a parametrized vector class mirroring that of PythonMultiMapping'''
    
    class Vector(Structure):
        _fields_ = (('outer', c_size_t),
                    ('inner', c_size_t),
                    ('data', POINTER(t)))

        def numpy(self):
            shape = self.outer, self.inner
            return np.ctypeslib.as_array(self.data, shape)
        
    return Vector
                    

class CompressedStorage(Structure):
    '''struct mirroring eigen sparse matrices internal storage'''
    _fields_ = (('values', POINTER(c_double)),
                ('indices', POINTER(c_int)),
                ('size', c_size_t),
                ('allocated_size', c_size_t))
                 

class SparseMatrix(Structure):
    '''struct mirroring eigen sparse matrices'''
    
    _fields_ = (('options', c_int), # warning: this is actually a bool (rvalue) + enum (options)
                ('outer_size', c_int),
                ('inner_size', c_int),
                ('outer_index', POINTER(c_int)),
                ('inner_nonzero', POINTER(c_int)),
                ('data', CompressedStorage))


    def _to_scipy(self):
        '''construct a scipy view of the eigen matrix

        warning: if the scipy view reallocates, it will no longer
        alias the eigen matrix. use the provided view context instead

        '''
        try:
            outer_index = np.ctypeslib.as_array(self.outer_index, (self.outer_size + 1,) )

            values = np.ctypeslib.as_array(self.data.values, (self.data.size,) )
            # array(self.data.values, self.data.size))
            inner_indices = np.ctypeslib.as_array(self.data.indices, (self.data.size,) )

            return sp.sparse.csr_matrix( (values, inner_indices, outer_index),
                                         (self.outer_size, self.inner_size))
        except (ValueError, AttributeError):
            # zero matrix: return empty view
            shape = (self.outer_size, self.inner_size)
            return sp.sparse.csr_matrix( shape )


    @staticmethod
    def from_scipy(s):
        '''construct a (fake) eigen sparse matrix using data pointers from a scipy sparse matrix

        use dll.eigen_sparse_matrix_assign to assign

        '''
        res = SparseMatrix()

        values, inner_indices, outer_index = s.data, s.indices, s.indptr

        res.options = 0    
        res.outer_size, res.inner_size = s.shape

        res.outer_index = outer_index.ctypes.data_as(POINTER(c_int))
        res.inner_nonzero = None

        res.data.values = values.ctypes.data_as(POINTER(c_double))
        res.data.indices = inner_indices.ctypes.data_as(POINTER(c_int))
        res.data.size = values.size
        res.data.allocated_size = values.size

        return res

    
    @contextmanager
    def view(self):
        '''a safe scipy view of an eigen sparse matrix.

        if the scipy view reallocates, the context exit will
        automatically reassign the view to the eigen sparse matrix.
        '''

        handle = self._to_scipy()
        
        try:
            yield handle
        finally:
            eig = SparseMatrix.from_scipy(handle)

            if not self.data.values or (addressof(self.data.values.contents) !=
                                        addressof(eig.data.values.contents)):
                dll.eigen_sparse_matrix_assign(pointer(self), pointer(eig))
                

# we need a few c functions
dll = CDLL(find_library('Compliant'))

dll.eigen_sparse_matrix_assign.restype = None
dll.eigen_sparse_matrix_assign.argtypes = (POINTER(SparseMatrix), POINTER(SparseMatrix))
    


def callback(restype, *args):
    '''a parametrized decorator that wraps a function into a c callback
    with given argument/return types

    '''
    
    def decorator(func):
        return CFUNCTYPE(restype, *args)(func)

    return decorator


@contextmanager
def nested(ctx):
    '''nested context manager: use all the contexts in a list in given
    order

    this is to help python2 with nested contexts and ease transition
    to python3

    '''
    if not ctx:
        yield ()
    else:
        with ctx[0] as h, nested(ctx[1:]) as t:
            yield (h, ) + t


def dofs_type(obj):
    '''return dofs ctype for a MechanicalObject'''
    _, _, name = obj.findData('position').getValueVoidPtr()
    return sofa_numpy.ctypeFromName[name]

            
class Mapping(object):
    '''a nice mapping wrapper class for PythonMultiMapping'''
    
    
    def __init__(self, node, **kwargs):
        '''you need to provide at least input/output kwargs'''
        
        self.node = node

        self.src = kwargs['input'] 
        self.dst = kwargs['output']

        # create wrapped mapping
        self.obj = node.createObject('PythonMultiMapping', **kwargs)

        # find source/dest scalar types
        source = self.obj.getFrom()
        in_type = dofs_type(source[0])
        assert all(in_type == dofs_type(s) for s in source[1:])
        
        destination = self.obj.getTo()
        assert len(destination) == 1
        out_type = dofs_type(destination[0])

        # setup callbacks
        @callback(None, Vec(out_type), POINTER(Vec(in_type)), c_size_t)
        def apply_callback(output, inputs, n):
            self.apply( output.numpy(), tuple(inputs[i].numpy() for i in range(n) ) )
            return

        @callback(None, POINTER( POINTER(SparseMatrix) ), POINTER(Vec(in_type)), c_size_t)
        def jacobian_callback(js, inputs, n):

            ctx = tuple( js[i].contents.view() for i in range(n) )
            inputs = tuple(inputs[i].numpy() for i in range(n) )

            with nested( ctx ) as js:
                self.jacobian(js, inputs)
                return 

        @callback(None, POINTER( SparseMatrix), POINTER(Vec(in_type)), c_size_t, Vec(out_type))
        def gs_callback(gs, inputs, n, force):

            inputs = tuple(inputs[i].numpy() for i in range(n) )
            
            with gs.contents.view() as gs:
                self.geometric_stiffness(gs, inputs, force.numpy())
                return 
            
        # set callbacks
        set_opaque(self.obj, 'apply_callback', apply_callback)
        set_opaque(self.obj, 'jacobian_callback', jacobian_callback)
        set_opaque(self.obj, 'gs_callback', gs_callback)                

        # keep a handle to avoid gc
        self.refs = [apply_callback, jacobian_callback, gs_callback]
        
        
    def apply(self, out, at):
        '''apply mapping to at putting result in out: out = self(at)

        at is a tuple of the position vectors for each of the mapping
        input states. 

        out is the position state vector for the mapping output state.

        '''
        
        pass

    def jacobian(self, js, at):
        '''build mapping jacobian blocks: js = J(at)

        at is a tuple of the position vectors for each of the mapping
        input states. 

        js is a tuple of jacobian blocks, one for each input state.

        '''

        pass

    def geometric_stiffness(self, gs, at, force):
        '''build geometric stiffness matrix: gs = dJ(at)^T * force

        gs is a sparse matrix with total size equal to the sum of all
        input dofs dimension

        force is the force state vector for mapping output state

        '''
        pass
    
