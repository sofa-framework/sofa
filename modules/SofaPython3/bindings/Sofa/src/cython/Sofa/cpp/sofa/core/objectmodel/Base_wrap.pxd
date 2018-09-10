# -*- coding: ASCII -*-
from libcpp.string cimport string as cpp_string
from .BaseData_wrap cimport BaseData as cpp_BaseData
from Sofa.cpp.sofa.helper.vector_wrap cimport vector as cpp_SofaVector

cdef extern from "<sofa/core/objectmodel/Base.h>" namespace "sofa::core::objectmodel":
    cdef cppclass Base:
        cpp_string getName()
        cpp_string getTypeName()
        cpp_string getClassName()
        cpp_string getTemplateName()

        cpp_BaseData* findData( cpp_string &name )
        cpp_SofaVector[cpp_BaseData*]& getDataFields()
