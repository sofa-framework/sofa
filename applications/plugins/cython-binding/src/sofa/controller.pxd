# -*- coding: ASCII -*-
from cpython.ref cimport PyObject as _PyObject
from boost cimport intrusive_ptr
from libcpp cimport bool
from libcpp.string cimport string as libcpp_string 

from sofa.baseobject cimport _BaseObject, _VisualParams

cdef extern from *:
    ctypedef _VisualParams* _constVisualParamsPtr "const sofa::core::visual::VisualParams*"
    
   
cdef extern from "cythoncontroller.cpp_inl" :
        cdef cppclass _CythonController "CythonController" (_BaseObject): 
                _PyObject* m_obj 
                void setPythonObject(_PyObject*) 

cdef extern from "../../../../../framework/sofa/core/ObjectFactory.h" namespace "sofa::core::":
        cdef cppclass _RegisterObject "sofa::core::RegisterObject":
                    _RegisterObject(libcpp_string& name)
                    
                    _RegisterObject& addMy "sofa::core::RegisterObject::add<PythonController>" ()
