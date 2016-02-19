# -*- coding: UTF8 -*
from base cimport Base, _Base

cdef extern from "../../../../../framework/sofa/core/objectmodel/BaseContext.h" namespace "sofa::core::objectmodel": 
    cdef cppclass _BaseContext "sofa::core::objectmodel::BaseContext" (_Base):        
        _BaseContext() except + 
  
  
cdef class BaseContext(Base):         
        cdef _BaseContext* basecontextptr 
       
        @staticmethod
        cdef createFrom(_BaseContext* aContext)
                
