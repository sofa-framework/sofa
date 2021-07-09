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
    'unsigned char': ctypes.c_ubyte,
    'short': ctypes.c_short,
    'unsigned short': ctypes.c_ushort,
    'int': ctypes.c_int,
    'unsigned int': ctypes.c_uint,
    'long': ctypes.c_long,
    'unsigned long': ctypes.c_ulong,
    'd': ctypes.c_double,
    'f': ctypes.c_float,
    'b': ctypes.c_char,
    'B': ctypes.c_ubyte,
    'h': ctypes.c_short,
    'H': ctypes.c_ushort,
    'i': ctypes.c_int,
    'I': ctypes.c_uint,
    'l': ctypes.c_long,
    'L': ctypes.c_ulong,
    'q': ctypes.c_longlong,
    'Q': ctypes.c_ulonglong,

}


def as_numpy( data ):
    '''maps data content as a numpy array'''

    ptr, shape, typename = data.getValueVoidPtr()

    type = ctypeFromName.get(typename,None)
    if not type: raise Exception("can't map data of type " + typename)

    # print (shape)

    # fold
    array_type = reduce(lambda x, y: x * y, reversed(shape), type)
    array = array_type.from_address(ptr)
    return numpy.ctypeslib.as_array(array)

    # https://github.com/numpy/numpy/issues/6511
    # array = ctypes.cast( ctypes.c_void_p(ptr), ctypes.POINTER(type))
    # return numpy.ctypeslib.as_array(array, shape)


# convenience
def numpy_data(obj, name):
    data = obj.findData(name)
    return as_numpy(data)


def vec_as_numpy( (ptr, size, typename) ):
    '''maps vec as a numpy array'''

    type = ctypeFromName.get(typename,None)
    if not type: raise Exception("can't map data of type " + typename)

    array_type = type * size
    array = array_type.from_address(ptr)
    return numpy.ctypeslib.as_array(array)
