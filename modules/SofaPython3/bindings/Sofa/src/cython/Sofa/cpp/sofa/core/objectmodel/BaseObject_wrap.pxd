# -*- coding: ASCII -*-
from Sofa.cpp.sofa.core.objectmodel.Base_wrap cimport Base as cpp_Base

cdef extern from "<sofa/core/objectmodel/BaseObject.h>" namespace "sofa::core::objectmodel":       
    cdef cppclass BaseObject(cpp_Base):
        BaseObject() except + 
        
        #Initialization method called at graph creation and modification, during top-down traversal.
        void init()

        #Initialization method called at graph creation and modification, during bottom-up traversal.
        void bwdInit()

        #Update method called when variables used in precomputation are modified.
        void reinit()

        #Reset to initial state
        void reset()
