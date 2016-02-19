# -*- coding: UTF8 -*
cimport libcpp.cast as cast
from basenode cimport _BaseNode
from libcpp.cast cimport dynamic_cast 
from libcpp.vector cimport vector

cdef extern from "": 
        _BaseNode* dynamic_cast_basenode_ptr "dynamic_cast< sofa::core::objectmodel::BaseNode* >" (_Base*) except NULL

cdef class BaseNode(Base):
        """ A Node is a class defining the main scene data structure of a simulation.
            It defined hierarchical relations between elements. Each node can have parent and child nodes 
            (potentially defining a tree), as well as attached objects (the leaves of the tree).
        """
        def getRoot(BaseNode self):
                return BaseNode.createFrom(self.basenodeptr.getRoot())
                
        @staticmethod
        cdef createFrom(_BaseNode* aNode):
                cdef BaseNode py_obj = BaseNode.__new__(BaseNode)
                py_obj.realptr = py_obj.basenodeptr = aNode 
                
                return py_obj 
     
        
        
