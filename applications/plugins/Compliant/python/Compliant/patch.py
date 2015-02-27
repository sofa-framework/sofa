
# stolen from
# http://stackoverflow.com/questions/6738987/extension-method-for-python-built-in-types

import ctypes
import Sofa

class PyObject(ctypes.Structure):
    pass

Py_ssize_t = hasattr(ctypes.pythonapi, 'Py_InitModule4_64') and ctypes.c_int64 or ctypes.c_int

PyObject._fields_ = [
    ('ob_refcnt', Py_ssize_t),
    ('ob_type', ctypes.POINTER(PyObject))
]

class SlotsPointer(PyObject):
    _fields_ = [('dict', ctypes.POINTER(PyObject))]


def class_dict( klass ):
    '''proxy for class __dict__ through CPython. not quite
    pythonic.'''

    name = klass.__name__
    slots = getattr(klass, '__dict__')

    pointer = SlotsPointer.from_address(id(slots))

    # not exactly sure why we need this dictionary but hey, it works
    namespace = {}

    ctypes.pythonapi.PyDict_SetItem(
        ctypes.py_object(namespace),
        ctypes.py_object(name),
        pointer.dict,
    )

    return namespace[name]
    

def sofa_class( cls ):
    '''decorator for sofa classes (with the same name)
    
    extends sofa classes with cls.__dict__ through CPython. not quite
    pythonic.

    '''
    
    klass = getattr(Sofa, cls.__name__)
    
    name = klass.__name__
    slots = getattr(klass, '__dict__')

    pointer = SlotsPointer.from_address(id(slots))

    # not exactly sure why we need this dictionary but hey, it works
    namespace = {}

    ctypes.pythonapi.PyDict_SetItem(
        ctypes.py_object(namespace),
        ctypes.py_object(name),
        pointer.dict,
    )

    # update class dictionnary
    namespace[name].update( cls.__dict__ )

    return cls


def instance(obj, cls):
    '''change class for obj. use wisely.'''
    
    ptr = PyObject.from_address( id(obj) )

    # don't forget to increase refcount lol
    ptr.ob_refcnt += 1
    ptr.ob_type = ctypes.pointer( PyObject.from_address( id(cls) ) )

    return obj

