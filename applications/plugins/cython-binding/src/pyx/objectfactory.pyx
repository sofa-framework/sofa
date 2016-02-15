# -*- coding: UTF8 -*
cimport libcpp.cast as cast
from libcpp.cast cimport dynamic_cast 

from baseobject cimport BaseObject as _BaseObject, SPtr as _BaseObjectSPtr
from baseobjectdescription cimport BaseObjectDescription as _BaseObjectDescription
from objectfactory cimport ObjectFactory as _ObjectFactory, CreateObject as _CreateObject

cdef class ObjectFactory:
        """An utilitary class allowing to create object attached to a specific location of the scene graph
        
           Examples:
                r = Simulation.getRoot()
                d = BaseObjectDescription("object name", "MechanicalObject") 
                c = ObjectFactory.createObject(r, d) 
                print("Object of type : "+str(c.getClassname())+" and name: "+str(c.getName())
        """

        @staticmethod
        def createObject(BaseContext aContext not None, BaseObjectDescription aDesc not None ):
                cdef _BaseObjectSPtr p = _CreateObject(aContext.basecontextptr, aDesc.baseobjectdescriptionptr)
                if p.get() == NULL :
                        raise Exception("Unable to create an object of type: "+str(aDesc.getName())) 
                return BaseObject.createFrom(p.get())
        
        
