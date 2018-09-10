# -*- coding: ASCII -*-
cimport libcpp.cast as cast
from libcpp.cast cimport dynamic_cast
from libcpp.vector cimport vector

from .cpp.sofa.core.objectmodel.Base_wrap cimport Base as _Base
from .cpp.sofa.core.objectmodel.BaseNode_wrap cimport BaseNode as _BaseNode

#cdef extern from "":
#        _BaseNode* dynamic_cast_basenode_ptr "dynamic_cast< sofa::core::objectmodel::BaseNode* >" (_Base*) except NULL

#cdef createFrom(_BaseNode* aNode):
#    cdef BaseNode py_obj = Node.__new__(Node)
#    py_obj.realptr = aNode
#    return py_obj

cdef class BaseNode(Base):

    def getRoot(BaseNode self):
        print("COUTOU")
        #        return BaseNode.createFrom(self.realptr.getRoot())



