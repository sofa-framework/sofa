# -*- coding: ASCII -*-
from .cpp.sofa.core.objectmodel.BaseData_wrap cimport BaseData as _BaseData

## Factory method to create a python BaseData wrapping around the C++ _BaseData pointer. 
cdef toPython(_BaseData* ptr)


cdef class BaseData:
        # In cython binding we assume that naked pointer are always valid and under the responsability of the C layer.
        cdef _BaseData* realptr

