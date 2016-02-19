# -*- coding: UTF8 -*
from libcpp cimport bool
from libcpp.string cimport string as libcpp_string
from sofa.base cimport Base, _Base
from sofa.baseobject cimport BaseObject, _SPtr as _BaseObjectSPtr
from cpython.object cimport PyObject_CallMethod
from cython.operator cimport dereference as deref, preincrement as inc, address as address
from sofa.basecontext cimport BaseContext

cdef cppclass PythonController "PythonController" (_CythonController):
        void draw(_constVisualParamsPtr p):
                if m_obj != NULL:
                        PyObject_CallMethod(<object>m_obj, "draw", NULL)
cdef extern from "": 
        PythonController* dynamic_cast_pythoncontroller_ptr "dynamic_cast< PythonController* >" (_Base*) except NULL
        _BaseObjectSPtr createAController "sofa::core::objectmodel::New<PythonController>" ()
        
        
cdef class Controller(BaseObject):
        cdef _BaseObjectSPtr controllerptr 

        def __init__(self, Base b not None):
                self.realptr = self.baseobjectptr =  dynamic_cast_pythoncontroller_ptr(b.realptr)
                self.controllerptr = _BaseObjectSPtr(dynamic_cast_pythoncontroller_ptr(b.realptr))
                if self.controllerptr.get() == NULL:
                        raise Exception("Unable to create this object from a broken python controller 1")
        
        @staticmethod
        cdef createFromPythonController(_BaseObjectSPtr b):
                cdef Controller self = Controller.__new__(Controller)
                self.controllerptr =  b
                self.realptr = b.get()
                self.baseobjectptr = b.get() 
        
                if self.controllerptr.get() == NULL:
                        raise Exception("Unable to create this object from a broken python controller 2")
                return self
                
        def setPythonObject(self, o):
                dynamic_cast_pythoncontroller_ptr(self.controllerptr.get()).setPythonObject(<_PyObject*>o)
        
cdef int myclassid = <int>_RegisterObject("PythonController").addMy()

def createObjectFromPython(BaseContext context not None, o not None):
        cdef _BaseObjectSPtr sptr = _BaseObjectSPtr(new PythonController())
        
        if sptr.get() == NULL:
                raise Exception("Unable to create an Object From Python")
        c = Controller.createFromPythonController( sptr )
        c.setPythonObject(o)
        return c
