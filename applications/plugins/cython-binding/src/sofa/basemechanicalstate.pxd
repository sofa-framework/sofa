# -*- coding: UTF8 -*
from libcpp cimport bool
from sofa.baseobject cimport BaseObject, _BaseObject
from libcpp.string cimport string as libcpp_string 

cdef extern from "../../../../../framework/sofa/core/behavior/BaseMechanicalState.h" namespace "sofa::core::behavior": 
      
    cdef cppclass _BaseMechanicalState "sofa::core::behavior::BaseMechanicalState" (_BaseObject):
        _BaseMechanicalState() except + 
        double getPX(size_t i) 
        double getPY(size_t i) 
        double getPZ(size_t i) 
        
        void applyScale(double sx, double sy, double sz) 
        void applyTranslation(double tx, double ty, double tz) 
        void applyRotation(double rx, double ry, double rz)
        
        void resize(size_t newsize)  
        int getSize() 
        
cdef class BaseMechanicalState(BaseObject):
        cdef _BaseMechanicalState* mechanicalstateptr 
        
        @staticmethod
        cdef createFrom(_BaseMechanicalState* aptr)
        
