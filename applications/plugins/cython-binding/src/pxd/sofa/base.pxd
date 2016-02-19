# -*- coding: ASCII -*-
from libcpp cimport bool
from libcpp.string cimport string as libcpp_string 

from sofa.sofavector cimport vector as sofavector
from sofa.basedata cimport _BaseData

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
        cdef createFrom(_Base* aBase)
        
