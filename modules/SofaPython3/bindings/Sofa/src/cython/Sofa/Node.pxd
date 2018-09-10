from Sofa.cpp.sofa.simulation.Node_wrap cimport Node as Node_cpp, SPtr as Node_SPtr


cdef create(str name)

cdef class Node:
        """ A Node is a class defining the main scene data structure of a simulation.
            It defined hierarchical relations between elements. Each node can have parent and child nodes 
            (potentially defining a tree), as well as attached objects (the leaves of the tree).
        """
        cdef Node_SPtr ptr
