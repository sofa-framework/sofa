# -*- coding: ASCII -*-

from .cpp.sofa.core.objectmodel.BaseNode_wrap cimport BaseNode as _BaseNode
from .Base cimport Base

cdef class BaseNode(Base):
        """ A Node is a class defining the main scene data structure of a simulation.
            It defined hierarchical relations between elements. Each node can have parent and child nodes
            (potentially defining a tree), as well as attached objects (the leaves of the tree).
        """
        #cdef _BaseNode* realptr



