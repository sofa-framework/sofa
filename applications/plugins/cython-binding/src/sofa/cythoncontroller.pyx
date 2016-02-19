# -*- coding: UTF8 -*
from libcpp cimport bool
from libcpp.string cimport string as libcpp_string
   
cdef cppclass _MyController "MyController" (_BaseObject): 
       cdef PyObject* obj
       void init():
                print("ZUT") 
       
       void setObject():
                print("SET THE CONTROLLER")
                       
cdef class Controller(object):
        cdef PyObject* controllerptr  
        def __init__(self):
                print("COUCOU")
        
        def getObject(self):
                return <object>self.controllerptr
                
        def setObject(self, o):
                self.controllerptr = <_PyObject*>o  
                
cdef int myclassid = <int>_RegisterObject("MyController").addMy()

