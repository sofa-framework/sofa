# -*- coding: ASCII -*-
from libcpp.cast cimport dynamic_cast 
from libcpp.vector cimport vector

from .cpp.sofa.simulation.Node_wrap cimport Node as _Node, SPtr as _NodeSPtr
from .cpp.sofa.core.objectmodel.BaseNode_wrap cimport BaseNode as _BaseNode
from .cpp.sofa.core.objectmodel.BaseObject_wrap cimport BaseObject as _BaseObject
from .BaseNode cimport BaseNode
from .BaseObject cimport BaseObject

#cdef extern from "":
#        _Node* dynamic_cast_basenode_ptr "dynamic_cast< sofa::core::objectmodel::Node* >" (_BaseNode*) except NULL

cdef wrapPythonAroundCPP(_NodeSPtr aNode):
        cdef Node py_obj = Node.__new__(Node)
        py_obj.ptr = aNode
        return py_obj

cdef wrapPythonAroundPTR(_Node* aNode):
            cdef Node py_obj = Node.__new__(Node)
            py_obj.ptr = _NodeSPtr(aNode)
            return py_obj

cdef create(str name):
    cdef Node py_obj = Node.__new__(Node)
    py_obj.ptr = _Node.create(name.encode("ASCII"))
    return py_obj

cdef class Node(BaseNode):
        """ A Node is a class defining the main scene data structure of a simulation.
            It defined hierarchical relations between elements. Each node can have parent and child nodes 
            (potentially defining a tree), as well as attached objects (the leaves of the tree).
        """

        def __cinit__(self):
                self.ptr = _NodeSPtr() #_Node.create(b"root")

        def __init__(self):
                raise Exception("Not creatable object")

        def getRoot(Node self):
                cdef _Node* n= <_Node*>self.ptr.get().getRoot()
                return wrapPythonAroundPTR( n )
                
        def getTreeNode(Node self not None, aName not None):
                """ Get a descendant node given its name"""
                assert isinstance(aName, (str)), 'arg aName has a wrong type. string is expected instead of '+str(type(aName))
                
                cdef _Node* n=self.ptr.get().getTreeNode(aName)
                if n != NULL:
                        return wrapPythonAroundPTR(n)
                return None
                
        def getChild(Node self not None, aName not None):
                """ Get a child node given its name """
                assert isinstance(aName, (str)), 'arg aName has a wrong type. string is expected instead of '+str(type(aName))
                
                cdef _Node* n=self.ptr.get().getChild(aName)
                if n != NULL:
                        return wrapPythonAroundPTR(n)
                return None
                
        def getObject(Node self not None, aName not None):
                """ Get an object attached to the current node given its name """
                assert isinstance(aName, (str)), 'arg aName has a wrong type. string is expected instead of '+str(type(aName))
                
                cdef _BaseObject* n=self.ptr.get().getObject(aName)
                if n != NULL:
                        return BaseObject.wrapPythonAroundCPP(n)
                return None
        
        def __str__(self):
                return "Node[{}]({})".format(self.ptr.get().getClassName().decode("ASCII"),
                                             self.ptr.get().getName().decode("ASCII"))

        def getName(self):
                return self.ptr.get().getName()
                
        def createChild(Node self not None, str aName not None):
                """ Create, add, then return the new child of this Node """

                return wrapPythonAroundCPP( self.ptr.get().createChild(aName.encode("ASCII")) )

                
