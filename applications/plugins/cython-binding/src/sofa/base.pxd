# -*- coding: ASCII -*-
from libcpp cimport bool
from boost cimport intrusive_ptr
from libcpp.string cimport string as libcpp_string 

from sofa.basedata cimport _BaseData
from sofa.sofavector cimport vector as sofavector

cdef extern from "" namespace "sofa::core::objectmodel::Base": 
        ctypedef intrusive_ptr[_Base] _SPtr "sofa::core::objectmodel::Base::SPtr"

cdef extern from "../../../../../framework/sofa/core/objectmodel/Base.h" namespace "sofa::core::objectmodel": 
    cdef cppclass _Base "sofa::core::objectmodel::Base":
        _Base() except + 
        libcpp_string getName()  
        libcpp_string getTypeName()  
        libcpp_string getClassName()  
        libcpp_string getTemplateName()  
        
        _BaseData* findData( libcpp_string &name ) 
        
        sofavector[_BaseData*]& getDataFields() 
        
cdef class Base:
        cdef _Base* realptr
        
        @staticmethod
        cdef createBaseFrom(_Base* aBase)
        
       
