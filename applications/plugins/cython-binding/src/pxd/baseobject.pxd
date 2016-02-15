# -*- coding: ASCII -*-
from libcpp cimport bool
from boost cimport intrusive_ptr
from libcpp.string cimport string as libcpp_string 

from base cimport Base

cdef extern from "" namespace "sofa::core::objectmodel::BaseObject": 
    ctypedef intrusive_ptr[BaseObject] BaseObjectSPtr
    ctypedef intrusive_ptr[BaseObject] SPtr

cdef extern from "../../../../../framework/sofa/core/objectmodel/BaseObject.h" namespace "sofa::core::objectmodel": 
      
    cdef cppclass BaseObject(Base):
        BaseObject() except + 
        
        #Initialization method called at graph creation and modification, during top-down traversal.
        void init()

        #Initialization method called at graph creation and modification, during bottom-up traversal.
        void bwdInit()

        #Update method called when variables used in precomputation are modified.
        void reinit()

        #Reset to initial state
        void reset()
        
        # 

