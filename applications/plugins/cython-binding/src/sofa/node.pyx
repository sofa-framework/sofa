# -*- coding: UTF8 -*
from libcpp.cast cimport dynamic_cast 
from libcpp.vector cimport vector
from sofa.objectfactory cimport _ObjectFactory, ObjectFactory 
from sofa.base cimport Base
from sofa.basecontext cimport BaseContext
from sofa.baseobject cimport BaseObject
from sofa.baseobjectdescription cimport BaseObjectDescription
from sofa.base cimport _Base 
from sofa.node cimport _Node, _SPtr as _NodeSPtr

cdef extern from "": 
        _Node* dynamic_cast_basenode_ptr "dynamic_cast< sofa::core::objectmodel::Node* >" (_Base*) except NULL

cdef class Node(BaseNode):
        """ A Node is a class defining the main scene data structure of a simulation.
            It defined hierarchical relations between elements. Each node can have parent and child nodes 
            (potentially defining a tree), as well as attached objects (the leaves of the tree).
            
            Examples:
                
                n = Simulation.getRoot()

                Node: 
                        n.getContext()
                        n.getTreeNode("childname")
                        n.getChild("childname") 
                        n.getObject("objectname")
                        n.createChild("newchildname")
                        str(n)
                
                BaseNode:
                        n.getRoot() 
                        
                Base:
                        n.findData("gravity")
                        n.getName()  
                        n.getTypeName()
                        n.getClassName()
                        n.getTemplateName()
                        n.gravity             # To acccess a data field
                        n.gravity[0] = [1,2,3]             # change a data field values  
                        
                        n1 == n2                 # To check if two pythons object are pointing to the same C object.  
        """
        def __init__(self, Node src not None):
                self.realptr = self.basenodeptr = self.nodeptr = src.nodeptr
        
        def getContext(Node self not None):
                """ Returns the context of this node """
                return BaseContext.createBaseContextFrom(self.nodeptr.getContext())

        def getRoot(Node self):
                return Node.createNodeFrom(<_Node*>self.nodeptr.getRoot())
                
        def getTreeNode(Node self not None, aName not None):
                """ Get a descendant node given its name"""
                assert isinstance(aName, (str)), 'arg aName has a wrong type. string is expected instead of '+str(type(aName))
                
                cdef _Node* n=self.nodeptr.getTreeNode(aName)
                if n != NULL:
                        return Node.createNodeFrom(n)
                return None
                
        def getChild(Node self not None, aName not None):
                """ Get a child node given its name """
                assert isinstance(aName, (str)), 'arg aName has a wrong type. string is expected instead of '+str(type(aName))
                
                cdef _Node* n=self.nodeptr.getChild(aName)
                if n != NULL:
                        return Node.createNodeFrom(n)
                return None
                
        def getObject(Node self not None, aName not None):
                """ Get an object attached to the current node given its name """
                assert isinstance(aName, (str)), 'arg aName has a wrong type. string is expected instead of '+str(type(aName))
                
                cdef _BaseObject* n=self.nodeptr.getObject(aName)
                if n != NULL:
                        return BaseObject.createBaseObjectFrom(n)
                return None
                
        def createChild(Node self not None, aName not None):
                """ Create, add, then return the new child of this Node """
                assert isinstance(aName, (str)), 'arg aName has a wrong type. string is expected instead of '+str(type(aName))
                
                cdef _NodeSPtr n=self.nodeptr.createChild(aName)
                if n.get() != NULL:
                        return Node.createNodeFrom(n.get())
                return None
        
        def createObject(Node self not None, aType not None, **kwargs):
                """ Create and add a new object to the current node """
                desc = BaseObjectDescription(aType, aType)
                if kwargs is not None:
                        for key, value in kwargs.iteritems():
                                desc.setAttribute(key, value)
                return ObjectFactory.createObject(self.getContext(), desc)
        
        @staticmethod
        cdef createNodeFrom(_Node* aNode):
                if aNode == NULL:
                        raise Exception("Invalid pointer")
                        
                cdef Node py_obj = Node.__new__(Node)
                super(Base, py_obj).__init__()
                
                py_obj.nodeptr = aNode 
                py_obj.basenodeptr = aNode 
                py_obj.realptr = aNode
                return py_obj 

        def __str__(self):
                return "Node["+self.nodeptr.getClassName()+"]("+self.nodeptr.getName()+")"
         
