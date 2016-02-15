# -*- coding: UTF8 -*
from libcpp cimport bool
from libcpp.string cimport string as libcpp_string 
from boost cimport intrusive_ptr
from basenode cimport BaseNode
from baseobject cimport BaseObject
from basecontext cimport BaseContext 

cdef extern from "" namespace "sofa::simulation::Node": 
    ctypedef intrusive_ptr[Node] SPtr

cdef extern from "../../../../../modules/sofa/simulation/common/Node.h" namespace "sofa::simulation": 
    cdef cppclass Node(BaseNode):        
        Node() except + 
        
        Node* getTreeNode(libcpp_string& name) 
        Node* getChild(libcpp_string& name)
        SPtr createChild(libcpp_string& nodeName)
        BaseObject* getObject(libcpp_string& name) 
        BaseContext* getContext() 

