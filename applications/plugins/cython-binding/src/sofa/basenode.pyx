# -*- coding: UTF8 -*
cimport libcpp.cast as cast
from libcpp.cast cimport dynamic_cast 
from libcpp.vector cimport vector
#from basenode cimport _BaseNode

cdef extern from "": 
        _BaseNode* dynamic_cast_basenode_ptr "dynamic_cast< sofa::core::objectmodel::BaseNode* >" (_Base*) except NULL

cdef class BaseNode(Base):
        """ A Node is a class defining the main scene data structure of a simulation.
            It defined hierarchical relations between elements. Each node can have parent and child nodes 
            (potentially defining a tree), as well as attached objects (the leaves of the tree).
        """
        def getRoot(BaseNode self):
                return BaseNode.createBaseNodeFrom(self.basenodeptr.getRoot())
        
        def __init__(self, BaseNode src not None):
                self.realptr = self.basenodeptr = src.basenodeptr 
                
        @staticmethod
        cdef createBaseNodeFrom(_BaseNode* aNode):
                cdef BaseNode py_obj = BaseNode.__new__(BaseNode)
                super(Base, py_obj).__init__()                
                py_obj.realptr = py_obj.basenodeptr = aNode 
                
                return py_obj 
     
        
        
