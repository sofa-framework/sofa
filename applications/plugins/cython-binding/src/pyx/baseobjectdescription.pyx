# -*- coding: UTF8 -*
cimport libcpp.cast as cast
from libcpp.string cimport string as libcpp_string 
from baseobjectdescription cimport BaseObjectDescription as _BaseObjectDescription

cdef class BaseObjectDescription:       
        cdef _BaseObjectDescription* baseobjectdescriptionptr

        def __dealloc__(self):
                del self.baseobjectdescriptionptr
                
        def __init__(self, aName, aType):
                 assert isinstance(aName, (str)), 'arg aName has a wrong type. string is expected instead of '+str(type(aName))
                 assert isinstance(aType, (str)), 'arg aType has a wrong type. string is expected instead of '+str(type(aName))
                 
                 self.baseobjectdescriptionptr = new _BaseObjectDescription(aName, aType) 
        
        def getName(self):
                return self.baseobjectdescriptionptr.getName() 
                 
        def setAttribute(self, aName, aValue):
                self.baseobjectdescriptionptr.setAttribute(<libcpp_string>aName, aValue)
