# -*- coding: UTF8 -*
cimport libcpp.cast as cast
from base cimport Base, _Base
from basecontext cimport _BaseContext

cdef class BaseContext(Base):
        """A Context contains values or pointers to variables and parameters shared
           by a group of objects, typically refering to the same simulated body.
           Derived classes can defined simple isolated contexts or more powerful
           hierarchical representations (scene-graphs), in which case the context also
           implements the BaseNode interface.
           """   
        @staticmethod
        cdef createFrom(_BaseContext* aContext):
                cdef BaseContext py_obj = BaseContext.__new__(BaseContext)
                py_obj.realptr = py_obj.basecontextptr = aContext 
                
                return py_obj 
        
        
