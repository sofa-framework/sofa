# -*- coding: UTF8 -*
cimport libcpp.cast as cast
from baseobject cimport BaseObject as _BaseObject
from libcpp.cast cimport dynamic_cast 
from libcpp.vector cimport vector

cdef extern from "": 
        _BaseObject* dynamic_cast_baseobject_ptr "dynamic_cast< sofa::core::objectmodel::BaseObject* >" (_Base*) except NULL

cdef class BaseObject(Base):
        """All sofa components implementing specific simulation behaviors are named 'objects' and 
           inherits from the BaseObject class. 
        """
        cdef _BaseObject* baseobjectptr 
    
        def init(self):
                self.baseobjectptr.init()

        def bwdInit(self):
                self.baseobjectptr.bwdInit()

        def reinit(self):
                self.baseobjectptr.reinit()

        def reset(self):
                self.baseobjectptr.reset()
        
        @staticmethod
        cdef createFrom(_BaseObject* aBaseObject):
                cdef BaseObject py_obj = BaseObject.__new__(BaseObject)
                py_obj.realptr = py_obj.baseobjectptr = aBaseObject 
                return py_obj 
    
        def __init__(self, Base src not None):
                cdef BaseObject pyobj = BaseObject.__new__(BaseObject)
                cdef _BaseObject* obj = dynamic_cast_baseobject_ptr(src.realptr)
                if obj == NULL:
                        raise TypeError("Unable to get a BaseObject from this Base pointer...maybe it is a BaseNode, BaseContext or something else")        
                self.realptr = self.baseobjectptr = obj                 
        
        def __str__(self):
                return "BaseObject["+self.baseobjectptr.getClassName()+"]("+self.baseobjectptr.getName()+")" 
                
     
