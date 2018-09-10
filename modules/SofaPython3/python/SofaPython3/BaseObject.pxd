# -*- coding: ASCII -*-
from .cpp.sofa.core.objectmodel.BaseObject_wrap cimport BaseObject as _BaseObject
from .Base cimport Base

cdef class BaseObject(Base):
    cdef _BaseObject* baseobjectptr

cdef wrapPythonAroundCPP(_BaseObject* aBaseObject)

