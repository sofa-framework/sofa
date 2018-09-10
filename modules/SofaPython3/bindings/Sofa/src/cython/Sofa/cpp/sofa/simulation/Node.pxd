# -*- coding: ASCII -*-

from libcpp.string cimport string as libcpp_string 
from SofaPython3.cpp.sofa.core.objectmodel.BaseObject cimport BaseObject as cpp_BaseObject

cdef extern from "<sofa/simulation/common/Node.h>" namespace "sofa::simulation": 
    cdef cppclass Node(BaseNode):        
        Node() except + 
        
        Node* getTreeNode(libcpp_string& name) 
        Node* getChild(libcpp_string& name)
        BaseObject* getObject(libcpp_string& name) 

        #SPtr createChild(libcpp_string& nodeName)
        
