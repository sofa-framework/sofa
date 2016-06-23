## @author Maxime Tournier & Matthieu Nesme

import Sofa
import ctypes
import numpy

# TODO add more basic types
# check that type sizes are equivalent with c++ sizes
# or even better, find a way to get the type without any string lookup
ctypeFromName = {
    'double': ctypes.c_double,
    'float': ctypes.c_float,
    'bool': ctypes.c_bool,
    'char': ctypes.c_char,
    'uchar': ctypes.c_ubyte,
    'short': ctypes.c_short,
    'ushort': ctypes.c_ushort,
    'int': ctypes.c_int,
    'uint': ctypes.c_uint,
    'long': ctypes.c_long,
    'ulong': ctypes.c_ulong,
}


def as_numpy( data ):
    '''maps data content as a numpy array'''

    ptr, shape, typename = data.getValueVoidPtr()

    type = ctypeFromName.get(typename,None)
    if not type: raise Exception("can't map data of type " + typename)

    array = ctypes.cast( ctypes.c_void_p(ptr), ctypes.POINTER(type))
    return numpy.ctypeslib.as_array(array, shape )


# convenience
def numpy_data(obj, name):
    data = obj.findData(name)
    return as_numpy(data)
