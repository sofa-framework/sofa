# -*- coding: ASCII -*-
from libcpp.string cimport string as libcpp_string 
cdef extern from "boost/intrusive_ptr.hpp" namespace "boost":
        cdef cppclass intrusive_ptr[T]:
                intrusive_ptr()
                intrusive_ptr(T*)
                T* get()
                void reset(T * r);

from SofaPython3.cpp.sofa.core.objectmodel.BaseObject_wrap cimport BaseObject
from SofaPython3.cpp.sofa.core.objectmodel.BaseNode_wrap cimport BaseNode

cdef extern from "" namespace "sofa::simulation::Node":
    ctypedef intrusive_ptr[Node] SPtr

cdef extern from "<sofa/simulation/Node.h>" namespace "sofa::simulation":

    cdef cppclass Node(BaseNode):        
        Node() except +
        
        Node* getTreeNode(libcpp_string& name) 
        Node* getChild(libcpp_string& name)
        BaseObject* getObject(libcpp_string& name) 

        SPtr createChild(libcpp_string&)
        
        @staticmethod
        SPtr create(libcpp_string& name)
