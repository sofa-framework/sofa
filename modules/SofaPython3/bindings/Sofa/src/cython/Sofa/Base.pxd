# -*- coding: ASCII -*-
from .cpp.sofa.core.objectmodel.Base_wrap cimport Base as _Base

cdef class Base:
    cdef _Base* realptr
