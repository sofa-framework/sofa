# -*- coding: UTF8 -*
from libcpp cimport bool
from sofa.base cimport Base, _Base
from sofa.baseobject cimport _BaseObject, _SPtr as _BaseObjectSPtr
from boost cimport intrusive_ptr

cdef extern from "../../../../../framework/sofa/core/objectmodel/BaseContext.h" namespace "sofa::core::objectmodel": 
    cdef cppclass _BaseContext "sofa::core::objectmodel::BaseContext" (_Base):        
        _BaseContext() except + 
        
        bool addObject(_BaseObjectSPtr)
        bool removeObject(_BaseObjectSPtr)
  
cdef class BaseContext(Base):         
        cdef _BaseContext* basecontextptr 
       
        @staticmethod
        cdef createBaseContextFrom(_BaseContext* aContext)
        
        cdef bool addObject(self, _BaseObjectSPtr)
        cdef bool removeObject(self, _BaseObjectSPtr)
