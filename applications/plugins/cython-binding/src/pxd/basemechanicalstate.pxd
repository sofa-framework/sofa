# -*- coding: UTF8 -*
from libcpp cimport bool
from baseobject cimport BaseObject
from libcpp.string cimport string as libcpp_string 

cdef extern from "../../../../../framework/sofa/core/behavior/BaseMechanicalState.h" namespace "sofa::core::behavior": 
      
    cdef cppclass BaseMechanicalState(BaseObject):
        BaseMechanicalState() except + 
        double getPX(size_t i) 
        double getPY(size_t i) 
        double getPZ(size_t i) 
        
        void applyScale(double sx, double sy, double sz) 
        void applyTranslation(double tx, double ty, double tz) 
        void applyRotation(double rx, double ry, double rz)
        
        void resize(size_t newsize)  
        int getSize() 
