
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

    
shapes = {
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
