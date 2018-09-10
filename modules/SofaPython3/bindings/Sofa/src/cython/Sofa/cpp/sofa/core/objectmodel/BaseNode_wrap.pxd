# -*- coding: ASCII -*-
from libcpp.string cimport string as libcpp_string 
from .Base_wrap cimport Base as cpp_Base

cdef extern from "<sofa/core/objectmodel/BaseNode.h>" namespace "sofa::core::objectmodel": 
   
    cdef cppclass BaseNode(cpp_Base):
        BaseNode() except + 
                
        BaseNode* getRoot()
        
        
                

