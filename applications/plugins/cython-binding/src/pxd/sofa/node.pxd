# -*- coding: UTF8 -*
from libcpp cimport bool
from libcpp.string cimport string as libcpp_string 
from boost cimport intrusive_ptr
from basenode cimport BaseNode, _BaseNode
from baseobject cimport _BaseObject
from basecontext cimport _BaseContext 

cdef extern from "" namespace "sofa::simulation::Node": 
    ctypedef intrusive_ptr[_Node] _SPtr "sofa::simulation::Node::SPtr"

cdef extern from "../../../../../modules/sofa/simulation/common/Node.h" namespace "sofa::simulation": 
    cdef cppclass _Node "sofa::simulation::Node" (_BaseNode):        
        _Node() except + 
        
        _Node* getTreeNode(libcpp_string& name) 
        _Node* getChild(libcpp_string& name)
        _SPtr createChild(libcpp_string& nodeName)
        _BaseObject* getObject(libcpp_string& name) 
        _BaseContext* getContext() 

cdef class Node(BaseNode):
        cdef _Node* nodeptr 
       
        @staticmethod
        cdef createFrom(_Node* aNode)
