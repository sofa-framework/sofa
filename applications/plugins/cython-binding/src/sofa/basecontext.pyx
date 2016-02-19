# -*- coding: UTF8 -*
cimport libcpp.cast as cast
from sofa.base cimport Base, _Base

cdef class BaseContext(Base):
        """A Context contains values or pointers to variables and parameters shared
           by a group of objects, typically refering to the same simulated body.
           Derived classes can defined simple isolated contexts or more powerful
           hierarchical representations (scene-graphs), in which case the context also
           implements the BaseNode interface.
           """   
        @staticmethod
        cdef createBaseContextFrom(_BaseContext* aContext):
                cdef BaseContext py_obj = BaseContext.__new__(BaseContext)
                py_obj.realptr = py_obj.basecontextptr = aContext 
                
                return py_obj 
        
        
        cdef bool addObject(self, _BaseObjectSPtr s):
                return self.basecontextptr.addObject(s)  
                
        cdef bool removeObject(self, _BaseObjectSPtr s):
                return self.basecontextptr.removeObject(s)                               
                
        
