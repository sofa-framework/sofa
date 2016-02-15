# -*- coding: ASCII -*-
from base cimport Base

cdef extern from "../../../../../framework/sofa/core/objectmodel/BaseNode.h" namespace "sofa::core::objectmodel": 
      
    cdef cppclass BaseNode(Base):
        BaseNode() except + 
        
        BaseNode* getRoot()        
        
              

