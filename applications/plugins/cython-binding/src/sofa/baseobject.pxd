# -*- coding: ASCII -*-
from libcpp cimport bool
from boost cimport intrusive_ptr
from libcpp.string cimport string as libcpp_string 

from sofa.base cimport Base, _Base

cdef extern from "../../../../../framework/sofa/core/visual/VisualParams.h" namespace "sofa::core::visual":
        cdef cppclass _VisualParams "sofa::core::visual::VisualParams":
                pass

cdef extern from "" namespace "sofa::core::objectmodel::BaseObject": 
    #ctypedef intrusive_ptr[_BaseObject] _BaseObjectSPtr
    ctypedef intrusive_ptr[_BaseObject] _SPtr "sofa::core::objectmodel::BaseObject::SPtr"

cdef extern from "../../../../../framework/sofa/core/objectmodel/BaseObject.h" namespace "sofa::core::objectmodel": 
    cdef cppclass _BaseObject "sofa::core::objectmodel::BaseObject" (_Base):
        _BaseObject() except + 
        
        #Initialization method called at graph creation and modification, during top-down traversal.
        void init()

        #Initialization method called at graph creation and modification, during bottom-up traversal.
        void bwdInit()

        #Update method called when variables used in precomputation are modified.
        void reinit()

        #Reset to initial state
        void reset()
        
        
cdef class BaseObject(Base):
        cdef _BaseObject* baseobjectptr 
        
        @staticmethod
        cdef createBaseObjectFrom(_BaseObject* aBaseObject)

