from __future__ import print_function, absolute_import
import Sofa

from ctypes import *
from ctypes.util import find_library

import numpy as np
import scipy as sp
from scipy import sparse

from SofaPython import SofaNumpy as sofa_numpy

from collections import namedtuple
from contextlib import contextmanager

dll_path = Sofa.loadPlugin('Compliant')
dll = CDLL(dll_path)


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
    '''a parametrized vector class for state vectors mirroring that of
    PythonMultiMapping
    '''
    
    class Vector(Structure):
        _fields_ = (('data', POINTER(t)),
                    ('outer', c_size_t),
                    ('inner', c_size_t),
                    )

        def numpy(self):
            return np.ctypeslib.as_array( as_buffer(self.data, self.outer, self.inner) )
        
    return Vector


def Eigen(t):
    '''a parametrized vector class for eigen vectors'''
    
    class Vector(Structure):
        _fields_ = (('data', POINTER(t)),
                    ('size', c_int))

        def numpy(self):
            return np.ctypeslib.as_array( as_buffer(self.data, self.size) )
        
    return Vector



class ScipyMatrix(Structure):
    '''all the data needed to alias a matrix in scipy'''
    
    _fields_ = (('rows', c_size_t),
                ('cols', c_size_t),
                ('outer_index', POINTER(c_int)),
                ('inner_nonzero', POINTER(c_int)),
                ('values', POINTER(c_double)),
                ('indices', POINTER(c_int)),
                ('size', c_size_t))


dll.eigen_sizeof.restype = c_size_t
dll.eigen_sizeof.argtypes = ()

def as_buffer(ptr, *size):
    '''cast a ctypes pointer as a correctly typed (multidimensional) array of given size

    this is needed to avoid a nasty memleak with numpy.ctypeslib.as_array
    '''
    
    addr = addressof(ptr.contents)
    
    buffer_type = type(ptr.contents)
    
    for s in reversed(size):
        buffer_type = buffer_type * s
        
    return buffer_type.from_address(addr)


class SparseMatrix(Structure):
    '''an opaque c type for eigen sparse matrices'''
    
    _fields_ = (('__bytes__', c_char * dll.eigen_sizeof() ), )
    
    def to_scipy(self):
        '''construct a scipy view of the eigen matrix

        warning: if the scipy view reallocates, it will no longer
        alias the eigen matrix. use the provided view context instead
        '''

        # fetch data from eigen matrix
        data = ScipyMatrix()
        dll.eigen_to_scipy(byref(data), byref(self))
        
        # needed: outer_index, data.values, data.size, data.indices, outer_size, inner_size
        outer_index = np.ctypeslib.as_array( as_buffer(data.outer_index, data.rows + 1) )
        
        shape = (data.rows, data.cols)

        if not data.values:
            return sp.sparse.csr_matrix( shape )

        values = np.ctypeslib.as_array( as_buffer(data.values, data.size) )
        inner_indices = np.ctypeslib.as_array( as_buffer(data.indices, data.size) )

        return sp.sparse.csr_matrix( (values, inner_indices, outer_index), shape)



    def from_scipy(self, s):
        '''assign eigen matrix from scipy matrix'''
        
        data = ScipyMatrix()
        values, inner_indices, outer_index = s.data, s.indices, s.indptr

        data.rows, data.cols = s.shape

        data.outer_index = outer_index.ctypes.data_as(POINTER(c_int))
        data.inner_nonzero = None

        data.values = values.ctypes.data_as(POINTER(c_double))
        data.indices = inner_indices.ctypes.data_as(POINTER(c_int))
        data.size = values.size

        dll.eigen_from_scipy(self, data)
        

    
    @contextmanager
    def view(self):
        '''a safe scipy view of an eigen sparse matrix.

        if the scipy view reallocates, the context exit will
        automatically reassign the view to the eigen sparse matrix.
        '''

        handle = self.to_scipy()
        data = handle.data.ctypes.data
        
        try:
            yield handle
        finally:
            new = handle.data.ctypes.data
            if new != data:
                # data pointer changed: assign back to eigen
                self.from_scipy(handle)
                

dll.eigen_to_scipy.restype = None
dll.eigen_to_scipy.argtypes = (POINTER(ScipyMatrix), POINTER(SparseMatrix))

dll.eigen_from_scipy.restype = None
dll.eigen_from_scipy.argtypes = (POINTER(SparseMatrix), POINTER(ScipyMatrix))


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


class Block(Structure):
    _fields_ = ( ('offset', c_size_t),
                 ('size', c_size_t),
                 ('_proj', c_void_p) )


class System(Structure):

    real = c_double

    _fields_ = (('m', c_uint),
                ('n', c_uint),
                ('dt', real),
                ('H', SparseMatrix),
                ('C', SparseMatrix),
                ('J', SparseMatrix),
                ('P', SparseMatrix),
                )

    class View(namedtuple('System', 'm n dt H C J P')): pass

    def size(self):
        return self.m + self.n
    
    @contextmanager
    def view(self):
        with self.H.view() as H, self.C.view() as C, self.J.view() as J, self.P.view() as P:
            yield System.View(m = self.m, n = self.n, dt = self.dt,
                       H = H, C = C, J = J, P = P)

class Solver(object):

    _instances_ = set()

    def release(self):
        del Solver._instances_[self]
    
    class Data(Structure):
        
        _fields_ = (('sys', POINTER(System)),
                    ('blocks', POINTER(Vec(Block))),
                    ('project', CFUNCTYPE(None, POINTER(System.real), POINTER(Block))))
        

        @contextmanager
        def view(self):
            with self.sys.contents.view() as sys:

                sys.blocks = lambda : (self.blocks.contents.data[i] for i in xrange(self.blocks.contents.outer))
                sys.project = lambda x, b: self.project(x.ctypes.data_as( POINTER(System.real)), b)
                
                yield sys
                
        
    def factor(self, view): pass
    
    def solve(self, res, view, rhs): pass

    def correct(self, res, view, rhs, damping):
        self.solve(res, view, rhs)
    
    def __init__(self, node):
        Solver._instances_.add(self)
        
        self.node = node
        self.obj = node.createObject('PythonSolver')

        vec = Eigen(System.real)

        @callback(None, POINTER(Solver.Data))
        def factor(data):
            with data.contents.view() as view:
                return self.factor(view)

        @callback(None, POINTER(vec), POINTER(Solver.Data), POINTER(vec))
        def solve(res, data, rhs):
            # TODO detect resizes for output vector?
            with data.contents.view() as view:
                return self.solve(res.contents.numpy(), view, rhs.contents.numpy())

        @callback(None, POINTER(vec), POINTER(Solver.Data), POINTER(vec), c_double)
        def correct(res, data, rhs, damping):
            # TODO detect resizes for output vector?
            with data.contents.view() as view:
                return self.correct(res.contents.numpy(), view, rhs.contents.numpy(), damping)

            
        set_opaque(self.obj, 'factor_callback', factor)
        set_opaque(self.obj, 'solve_callback', solve)
        set_opaque(self.obj, 'correct_callback', correct)                

        # keep refs to prevent gc
        self.handles = [factor, solve, correct]
        

class Mapping(object):
    '''a nice mapping wrapper class for PythonMultiMapping'''

    _instances_ = set()

    def release(self):
        del Mapping._instances_[self]


    def set_callback(self, name, cb):
        set_opaque(self.obj, name, cb)
        self._refs_[name] = cb

    def __del__(self):
        for k, v in self._refs_.iteritems():
            set_opaque(self.obj, k, type(v)(None))

        
    def __init__(self, node, **kwargs):
        '''you need to provide at least input/output kwargs'''

        # note: you need to 'release' to allow gc
        Mapping._instances_.add(self)
        
        self.node = node

        self.input = list(kwargs['input'])
        self.output = kwargs['output']

        in_templates = set()
        for i in self.input:
            in_templates.add(i.getTemplateName())

        if len(in_templates) != 1: 
            raise Exception('input dofs must have the same template')

        template = '{0},{1}'.format( next(iter(in_templates)), self.output.getTemplateName() )
        
        input = ' '.join( [x.getLinkPath() for x in self.input] )
        output = self.output.getLinkPath()
        
        # create wrapped mapping
        self.obj = node.createObject('PythonMultiMapping',
                                     input = input,
                                     output = output,
                                     template = template)
        
        # find source/dest scalar types
        source = self.obj.getFrom()
        in_type = dofs_type(source[0])
        assert all(in_type == dofs_type(s) for s in source[1:])
        
        destination = self.obj.getTo()
        assert len(destination) == 1
        out_type = dofs_type(destination[0])

        # derived type
        cls = type(self)

        # keep a handle to avoid gc
        self._refs_ = {}
                
        # setup callbacks
        if cls.apply is not Mapping.apply:
            
            @callback(None, POINTER(Vec(out_type)), POINTER(Vec(in_type)), c_size_t)
            def cb(output, inputs, n):
                self.apply( output.contents.numpy(), tuple(inputs[i].numpy() for i in range(n) ) )
                return
            
            self.set_callback('apply_callback', cb)

        if cls.jacobian is not Mapping.jacobian:

            @callback(None, POINTER( POINTER(SparseMatrix) ), POINTER(Vec(in_type)), c_size_t)
            def cb(js, inputs, n):
                ctx = tuple( js[i].contents.view() for i in xrange(n) )
                inputs = tuple(inputs[i].numpy() for i in xrange(n) )

                with nested( ctx ) as js:
                    self.jacobian(js, inputs)
                    return

            self.set_callback('jacobian_callback', cb)


        if cls.geometric_stiffness is not Mapping.geometric_stiffness:
            
            @callback(None, POINTER( SparseMatrix), POINTER(Vec(in_type)), c_size_t, POINTER(Vec(out_type)))
            def cb(gs, inputs, n, force):

                inputs = tuple(inputs[i].numpy() for i in range(n) )

                with gs.contents.view() as gs:
                    self.geometric_stiffness(gs, inputs, force.contents.numpy())
                    return 

            self.set_callback('gs_callback', cb)


        if cls.draw is not Mapping.draw:

            @callback(None)
            def cb():
                self.draw()
                
            self.set_callback('draw_callback', cb)

        
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
    

    def draw(self):
        '''draw mapping info for debugging purpose'''
        pass
