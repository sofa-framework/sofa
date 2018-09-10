# -*- coding: ASCII -*-
from libcpp.cast cimport dynamic_cast 
from libcpp.vector cimport vector

from .Base cimport Base
from .cpp.sofa.core.objectmodel.Base_wrap cimport Base as _Base
from .cpp.sofa.core.objectmodel.BaseObject_wrap cimport BaseObject as _BaseObject

cdef extern from "":
        _BaseObject* dynamic_cast_baseobject_ptr "dynamic_cast< sofa::core::objectmodel::BaseObject* >" (_Base*) except NULL

cdef wrapPythonAroundCPP(_BaseObject* aBaseObject):
        cdef BaseObject py_obj = BaseObject.__new__(BaseObject)
        py_obj.realptr = py_obj.baseobjectptr = aBaseObject
        return py_obj

cdef class BaseObject(Base):
        """All sofa simulation behaviors are named 'objects' and inherits from this BaseObject class.
        """    
        def __init__(self, Base src not None):
                cdef BaseObject pyobj = BaseObject.__new__(BaseObject)
                cdef _BaseObject* obj = dynamic_cast_baseobject_ptr(src.realptr)
                if obj == NULL:
                        raise TypeError("Unable to get a BaseObject from this Base pointer...maybe it is a BaseNode, BaseContext or something else")        
                self.realptr = self.baseobjectptr = obj                 
        
        def __str__(self):
                return "BaseObject["+self.baseobjectptr.getClassName()+"]("+self.baseobjectptr.getName()+")" 
                
        def init(self):
                self.baseobjectptr.init()

        def bwdInit(self):
                self.baseobjectptr.bwdInit()

        def reinit(self):
                self.baseobjectptr.reinit()

        def reset(self):
                self.baseobjectptr.reset()
        
        
