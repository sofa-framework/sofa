import Sofa

import numpy as np
import scipy as sp
from scipy import sparse
from contextlib import contextmanager

from ctypes import *

dll_path = Sofa.loadPlugin('SofaPython')
dll = CDLL(dll_path)



def matrix(dtype):
    '''matrix type constructor from scalar type (c_double or c_float)'''
    
    from_eigen_table = {
        c_double: dll.eigen_to_scipy_double,
        c_float: dll.eigen_to_scipy_float,        
    }


    to_eigen_table = {
        c_double: dll.eigen_from_scipy_double,
        c_float: dll.eigen_from_scipy_float,        
    }
    

    # note: we don't alias eigen matrices directly as the memory layout could change
    # between versions (it did once in the past IIRC), so we use a low-level
    # exchange data structure instead
    class Matrix(Structure):
        '''all the data needed to alias a sparse matrix in eigen/scipy (see ctypes.cpp)'''

        _fields_ = (('rows', c_size_t),
                    ('cols', c_size_t),
                    ('outer_index', POINTER(c_int)),
                    ('inner_nonzero', POINTER(c_int)),
                    ('values', POINTER(dtype)),
                    ('indices', POINTER(c_int)),
                    ('size', c_size_t))


        @staticmethod
        def from_scipy(s):
            '''build a sparse matrix alias from a scipy matrix'''            
            data = Matrix()
            values, inner_indices, outer_index = s.data, s.indices, s.indptr

            data.rows, data.cols = s.shape

            data.outer_index = outer_index.ctypes.data_as(POINTER(c_int))
            data.inner_nonzero = None

            # TODO mettre le bon type
            data.values = values.ctypes.data_as(POINTER(dtype))
            data.indices = inner_indices.ctypes.data_as(POINTER(c_int))
            data.size = values.size

            return data



        @staticmethod
        def from_eigen(ptr):
            '''build a sparse matrix alias from a pointer to an eigen matrix'''
            data = Matrix()
            from_eigen_table[dtype](byref(data), ptr)
            return data



        def to_eigen(self, ptr):
            '''assign current aliased matrix to the given eigen matrix'''
            return to_eigen_table[dtype](ptr, byref(self))
        

        def to_scipy(self, writeable = False):
            '''construct a scipy view of the current aliased matrix

            warning: if writable is True and the scipy view reallocates, it will
            no longer alias the sparse matrix. use the provided context instead
            of using this function directly
            '''

            data = self

            # needed: outer_index, data.values, data.size, data.indices, outer_size, inner_size
            outer_index = np.ctypeslib.as_array( as_buffer(data.outer_index, data.rows + 1) )

            shape = (data.rows, data.cols)

            if not data.values:
                return sp.sparse.csr_matrix( shape )

            values = np.ctypeslib.as_array( as_buffer(data.values, data.size) )
            if not writeable: values.flags['WRITEABLE'] = writable
            
            inner_indices = np.ctypeslib.as_array( as_buffer(data.indices, data.size) )

            return sp.sparse.csr_matrix( (values, inner_indices, outer_index), shape)


        @staticmethod
        @contextmanager
        def view(ptr):
            '''a context that provides a scipy view of an eigen matrix, assigning data back
            when modified on context exit. 
            '''
            
            view = Matrix.from_eigen(ptr).to_scipy(writeable = True)
            data = view.data.ctypes.data

            try:
                yield view
            finally:
                new = view.data.ctypes.data
                if new != data:
                    # data pointer changed: rebuild view and assign back to eigen
                    Matrix.from_scipy(view).to_eigen(ptr)

                # make sure that leaked handles are not writable
                view.data.flags['WRITEABLE'] = False

    return Matrix


Matrixd = matrix(c_double)
Matrixf = matrix(c_float)

# sparse <- eigen        
dll.eigen_to_scipy_double.restype = None
dll.eigen_to_scipy_double.argtypes = (POINTER(Matrixd), c_void_p)

dll.eigen_to_scipy_float.restype = None
dll.eigen_to_scipy_float.argtypes = (POINTER(Matrixf), c_void_p)

# eigen <- sparse
dll.eigen_from_scipy_double.restype = None
dll.eigen_from_scipy_double.argtypes = (c_void_p, POINTER(Matrixd))

dll.eigen_from_scipy_float.restype = None
dll.eigen_from_scipy_float.argtypes = (c_void_p, POINTER(Matrixf))


matrix_type_from_data_type = {
    'EigenBaseSparseMatrixd': Matrixd,
    'EigenBaseSparseMatrixf': Matrixf,    
}


@contextmanager
def data_view(obj, data_name):

    data = obj.findData(data_name)
    ptr, _, data_type = data.getValueVoidPtr()

    matrix_type = matrix_type_from_data_type[data_type]
    
    with matrix_type.view(ptr) as view:
        yield view


    
def as_buffer(ptr, *size):
    '''cast a ctypes pointer to a multidimensional array of given sizes'''
    
    addr = addressof(ptr.contents)
    
    buffer_type = type(ptr.contents)
    
    for s in reversed(size):
        buffer_type = buffer_type * s
        
    return buffer_type.from_address(addr)

