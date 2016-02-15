# -*- coding: ASCII -*-
from libcpp cimport bool
from libcpp.string cimport string as libcpp_string 

from sofavector cimport vector as sofavector
from basedata cimport BaseData

cdef extern from "../../../../../framework/sofa/core/objectmodel/Base.h" namespace "sofa::core::objectmodel": 
    cdef cppclass Base:
        Base() except + 
        libcpp_string getName()  
        libcpp_string getTypeName()  
        libcpp_string getClassName()  
        libcpp_string getTemplateName()  
        
        BaseData* findData( libcpp_string &name ) 
        
        sofavector[BaseData*]& getDataFields() 
