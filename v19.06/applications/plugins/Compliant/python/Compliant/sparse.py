import Sofa

import numpy as np
import scipy as sp
from scipy import sparse
from contextlib import contextmanager

from ctypes import *

dll_path = Sofa.loadPlugin('Compliant')
dll = CDLL(dll_path)


# note: we don't alias eigen matrices directly as the memory layout could change
# between versions (it did once in the past IIRC), so we use a low-level
# exchange data structure instead

class Matrix(Structure):
    '''all the data needed to alias a sparse matrix in eigen/scipy'''
    
    _fields_ = (('rows', c_size_t),
                ('cols', c_size_t),
                ('outer_index', POINTER(c_int)),
                ('inner_nonzero', POINTER(c_int)),
                ('values', POINTER(c_double)),
                ('indices', POINTER(c_int)),
                ('size', c_size_t))


    @staticmethod
    def from_scipy(s):
        data = Matrix()
        values, inner_indices, outer_index = s.data, s.indices, s.indptr
        
        data.rows, data.cols = s.shape

        data.outer_index = outer_index.ctypes.data_as(POINTER(c_int))
        data.inner_nonzero = None

        data.values = values.ctypes.data_as(POINTER(c_double))
        data.indices = inner_indices.ctypes.data_as(POINTER(c_int))
        data.size = values.size

        return data

    @staticmethod
    def from_eigen(ptr):
        data = Matrix()
        dll.eigen_to_scipy(byref(data), ptr)
        return data
        

    def to_eigen(self, ptr):
        return dll.eigen_from_scipy(ptr, byref(self))

    
    def to_scipy(self):
        '''warning: if the scipy view reallocates, it will no longer alias the sparse
        matrix.
        '''

        data = self
        
        # needed: outer_index, data.values, data.size, data.indices, outer_size, inner_size
        outer_index = np.ctypeslib.as_array( as_buffer(data.outer_index, data.rows + 1) )
        
        shape = (data.rows, data.cols)

        if not data.values:
            return sp.sparse.csr_matrix( shape )

        values = np.ctypeslib.as_array( as_buffer(data.values, data.size) )
        inner_indices = np.ctypeslib.as_array( as_buffer(data.indices, data.size) )

        return sp.sparse.csr_matrix( (values, inner_indices, outer_index), shape)


    @staticmethod
    @contextmanager
    def view(ptr):
        view = Matrix.from_eigen(ptr).to_scipy()
        data = view.data.ctypes.data
        
        try:
            yield view
        finally:
            new = view.data.ctypes.data
            if new != data:
                # data pointer changed: rebuild view and assign back to eigen
                Matrix.from_scipy(view).to_eigen(ptr)

            
# sparse <- eigen        
dll.eigen_to_scipy.restype = None
dll.eigen_to_scipy.argtypes = (POINTER(Matrix), c_void_p)

# eigen <- sparse
dll.eigen_from_scipy.restype = None
dll.eigen_from_scipy.argtypes = (c_void_p, POINTER(Matrix))
        
    
def as_buffer(ptr, *size):
    '''cast a ctypes pointer to a multidimensional array of given sizes'''
    
    addr = addressof(ptr.contents)
    
    buffer_type = type(ptr.contents)
    
    for s in reversed(size):
        buffer_type = buffer_type * s
        
    return buffer_type.from_address(addr)

