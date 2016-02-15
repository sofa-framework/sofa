# -*- coding: UTF8 -*
from base cimport Base

cdef extern from "../../../../../framework/sofa/core/objectmodel/BaseContext.h" namespace "sofa::core::objectmodel": 
    cdef cppclass BaseContext(Base):        
        BaseContext() except + 
        
        
