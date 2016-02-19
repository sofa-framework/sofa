# -*- coding: ASCII -*-
from sofa.base cimport Base, _Base

cdef extern from "../../../../../framework/sofa/core/objectmodel/BaseNode.h" namespace "sofa::core::objectmodel": 
      
    cdef cppclass _BaseNode "sofa::core::objectmodel::BaseNode" (_Base):
        _BaseNode() except + 
        
        _BaseNode* getRoot()        
        
              
cdef class BaseNode(Base):
        cdef _BaseNode* basenodeptr 
                     
        @staticmethod
        cdef createBaseNodeFrom(_BaseNode* aNode)

