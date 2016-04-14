
import ctypes

class PyObject(ctypes.Structure):
    '''ctypes representation of PyObject_HEAD'''
    
    _fields_ = (
        ('ob_refcnt', ctypes.c_size_t),
        ('ob_type', ctypes.c_void_p),
    )

class SofaPySPtr(ctypes.Structure):
    '''ctypes representation of sofa PySPtr'''
    _fields_ = (
        ('head', PyObject),
        ('obj', ctypes.c_void_p),
    )

class DataPointer(ctypes.Structure):
    '''result type for c function get_data_pointer'''
    _fields_ = (
        ('ptr', ctypes.c_void_p),
        ('size', ctypes.c_uint),
    )
    
    
def sofa_pointer( obj ):
    '''returns pointer to wrapped sofa object as a ctype.c_void_p'''

    # obj -> pyobject*
    pyobj = ctypes.c_void_p(id(obj))

    # pyobject* -> pysptr
    pysptr = ctypes.cast( pyobj, ctypes.POINTER(SofaPySPtr)).contents

    # obtain wrapped object
    sofaobj = pysptr.obj
    
    return sofaobj


def load_dll():
    '''load symbols from compliant dll. you need to load compliant plugin first.'''

    # TODO we probably need to make this portable
    global dll
    dll = ctypes.CDLL('libCompliant.so')

    global get_data_pointer
    get_data_pointer = dll.get_data_pointer
    
    get_data_pointer.restype = DataPointer
    get_data_pointer.argtypes = (ctypes.c_void_p, )

    global py_callback_type
    py_callback_type = ctypes.CFUNCTYPE(None)

    global set_py_callback
    set_py_callback = dll.set_py_callback
    
    set_py_callback.restype = None
    set_py_callback.argtypes = (ctypes.c_void_p, py_callback_type)
    
    
shapes = {
    'vector<double>': (ctypes.c_double, 1),    
    'vector<Vec1d>': (ctypes.c_double, 1),
    'vector<Vec3d>': (ctypes.c_double, 3),
    'vector<Vec6d>': (ctypes.c_double, 6),
    'vector<Rigid3dTypes::Coord>': (ctypes.c_double, 7),
    'vector<Rigid3dTypes::Deriv>': (ctypes.c_double, 6),    
}


import numpy
from numpy import ctypeslib

def as_numpy( data ):
    '''maps data content as a numpy array'''
    try:
        dll
    except NameError:
        load_dll()

    # TODO check type(data)
        
    ts = data.getValueTypeString()
    shape = shapes.get(ts, None)
    if not shape: raise Exception("can't map data of type " + ts)
    
    t, cols = shape
    
    sp = sofa_pointer(data)
    d = get_data_pointer(sp)
    rows = d.size
    
    array = ctypes.cast( ctypes.c_void_p(d.ptr), ctypes.POINTER(t))
    return ctypeslib.as_array(array, (rows, cols))

# convenience
def numpy_data(obj, name):
    data = obj.findData(name)
    return as_numpy(data)





class Mapping(object):
    '''wraps a PythonMultiMapping into something more usable'''
    
    def __init__(self, node, **kwargs):
        self.node = node

        self.src = kwargs['input'] 
        self.dst = kwargs['output']

        kwargs['template'] = '{0},{1}'.format( self.src[0].getTemplateName(),
                                               self.dst.getTemplateName())
        kwargs['input'] = multi_mapping_input(node, *self.src)
        kwargs['output'] = object_link_relative(node, self.dst)
        
        self.obj = node.createObject('PythonMultiMapping', **kwargs)
        
        self.resize()

        # callback
        # keep a handle on the closure to prevent gc
        self._update = lambda: self.update()
        
        set_py_callback( sofa_pointer(self.obj),
                         py_callback_type( self._update ) )
                         
    def update(self):
        pass
    

    def resize(self):
        '''update stuff after input/output resize'''

        pos = 'position'        
        vel = 'velocity'

        self.out_vel = numpy_data(self.dst, vel)
        self.out_pos = numpy_data(self.dst, pos)

        self.in_vel = [ numpy_data(s, vel) for s in self.src ]
        self.in_pos = [ numpy_data(s, pos) for s in self.src ]
        
        # out dim
        self.m, out_dim = self.out_vel.shape
        
        # in dim
        self.n = sum( v.shape[0] for v in self.in_vel  )
        in_dim = self.in_vel[0].shape[1]
        
        # resize jacobian/value data
        size = self.m * self.n * in_dim * out_dim
        
        self.obj.jacobian = ' '.join( ['0'] * size )
        self.obj.value = ' '.join( ['0'] * self.m * out_dim )
    
        # map numpy arrays
        self._jacobian = numpy_data(self.obj, 'jacobian')
        self._value = numpy_data(self.obj, 'value')

        # reshape jacobian
        self._jacobian = self._jacobian.reshape(self.m * out_dim, self.n * in_dim)

        
    @property
    def jacobian(self):
        '''a view of the jacobian matrix'''
        return self._jacobian

    @property
    def value(self):
        '''a view of the value vector'''
        return self._value

    # convenience
    @jacobian.setter
    def jacobian(self, value):
        self._jacobian[:] = value
    
    @value.setter
    def value(self, value):
        self._value[:] = value


from Tools import node_path_rel as node_path_relative

def object_link_relative(node, obj):
    return '@{0}/{1}'.format( node_path_relative(node, obj.getContext() ), obj.name )

def multi_mapping_input(node, *dofs ):
    return ' '.join( object_link_relative(node, x) for x in dofs )
