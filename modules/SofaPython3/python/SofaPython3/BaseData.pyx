# -*- coding: ASCII -*-
import cython

from libcpp cimport bool
from libc.math cimport floor
from .cpp.sofa.core.objectmodel.BaseData_wrap cimport BaseData as _BaseData
from .cpp.sofa.core.objectmodel.BaseData_wrap cimport AbstractTypeInfo as _AbstractTypeInfo

cdef toPython(_BaseData* ptr):
    cdef BaseData d = BaseData()
    d.realptr = ptr
    return d

cdef class BaseData(object):
        def __init__(self):
            raise TypeError("Cannot instanciate an empty BaseData")

        #def __getattr__(self, item):
        #        if item == "__members__":
        #                print("\n\n Help about: "self.help())
        #        raise AttributeError("has no attribute '"+item+"'")
        def __len__(self):
                cdef const _AbstractTypeInfo* nfo=self.realptr.getValueTypeInfo()
                cdef const void* ptr = self.realptr.getValueVoidPtr()
                return <long>(nfo.size(ptr)/nfo.size())
        
        def dim(self):
                cdef const _AbstractTypeInfo* nfo=self.realptr.getValueTypeInfo()
                cdef const void* ptr = self.realptr.getValueVoidPtr()
                return (nfo.size(ptr)/nfo.size(), nfo.size())
        
        def setSize(self, size):
                assert isinstance(size, (int)), 'size should be of integer type'
                cdef const _AbstractTypeInfo* nfo=self.realptr.getValueTypeInfo()
                cdef void* ptr = self.realptr.beginEditVoidPtr()
                cdef rowwidth = nfo.size() 
                nfo.setSize(ptr, rowwidth*size) 
                self.realptr.endEditVoidPtr()                     
                return len(self)
        
        
        def __getitem__(self, key):
                cdef const _AbstractTypeInfo* nfo=self.realptr.getValueTypeInfo()
                cdef const void* ptr = self.realptr.getValueVoidPtr()
                cdef rowwidth = nfo.size() 
                cdef numrow = nfo.size(ptr) / rowwidth 
                if not isinstance(key, int):
                        raise TypeError("Expecting the key to be an interger while the provided value ["+str(key)+"] seems to be of type "+str(type(key)))         
                if key>=numrow:
                        raise IndexError("Key ["+str(key)+"] is to big for this array of size "+str(numrow) )
                
                py_result = []
                for i in range(0, rowwidth):
                        if nfo.Scalar():
                                py_result.append(nfo.getScalarValue(self.realptr.getValueVoidPtr(), key*rowwidth+i))
                        if nfo.Integer():
                                py_result.append(nfo.getIntegerValue(self.realptr.getValueVoidPtr(), key*rowwidth+i))
                        if nfo.Text():
                                py_result.append(nfo.getTextValue(self.realptr.getValueVoidPtr(), key*rowwidth+i))
                
                if nfo.Container():
                        return py_result
                return py_result[0]
                
        def __setitem__(self, key, value):
                cdef const _AbstractTypeInfo* nfo=self.realptr.getValueTypeInfo()
                cdef void* ptr = self.realptr.beginEditVoidPtr()
                cdef rowwidth = nfo.size() 
                cdef numrow = nfo.size(ptr) / rowwidth 
                if not isinstance(key, int):
                        self.realptr.endEditVoidPtr()
                        raise TypeError("Expecting the key to be an interger while the provided value ["+str(key)+"] seems to be of type "+str(type(key)))         
                if key>=numrow:
                        self.realptr.endEditVoidPtr()
                        raise IndexError("Key ["+str(key)+"] is to big for this array of size "+str(rowwidth) )
                
                if nfo.Text():
                        nfo.setTextValue(ptr, key, value) 
                        self.realptr.endEditVoidPtr()
                        return
               
                        
                if len(value) != rowwidth:
                        self.realptr.endEditVoidPtr()
                        raise IndexError("Value has a length of size ["+str(len(value))+"] which differ from the expected size "+str(rowwidth) )
             
                for i in range(0, rowwidth):
                        if nfo.Scalar():
                                nfo.setScalarValue(ptr, key*rowwidth+i, value[i])
                        if nfo.Integer():
                                nfo.setIntegerValue(ptr, key*rowwidth+i, value[i])
                self.realptr.endEditVoidPtr()
        
        def __str__(self):
                return self.realptr.getValueString()
        
        def append(self, value):
                cdef const _AbstractTypeInfo* nfo=self.realptr.getValueTypeInfo()
                cdef void* ptr = self.realptr.beginEditVoidPtr()
                
                cdef rowwidth = nfo.size() 
                cdef numrow = nfo.size(ptr) / rowwidth 
                oldsize = nfo.size(ptr)
                                
                if nfo.Container() and not nfo.FixedSize():
                        nfo.setSize(ptr, nfo.size(ptr)+nfo.size())
                        
                        if rowwidth == 1:
                                if nfo.Scalar():
                                        nfo.setScalarValue(ptr, oldsize, value)
                                if nfo.Integer():
                                        nfo.setIntegerValue(ptr, oldsize, value) 
                        else:                               
                                for i in range(0, rowwidth):
                                        if nfo.Scalar():
                                                nfo.setScalarValue(ptr, oldsize+i, value[i])
                                        if nfo.Integer():
                                                nfo.setIntegerValue(ptr, oldsize+i, value[i])
                
                        self.realptr.endEditVoidPtr()
                        return
                self.realptr.endEditVoidPtr()
                raise TypeError("This DataField is not a container or it cannot be resized.")         
                        
        def getName(self):
                return self.realptr.getName()          

        def getValueString(self):
                return self.realptr.getValueString() 

        def getValueTypeString(self):
                return self.realptr.getValueTypeString() 
                
        def setPersistent(self, bool b):
                """ By default BaseData are not serialized. You can use this function
                    to indicate the contrary.  
                     
                    This may be usefull if you are implementing editting/modelling features
                    using the script and want the changed made to the scene to be saved.          
                """
                self.realptr.setPersistent(<bool>b)   
        
        def isPersistent(self):
                """ By default BaseData are not serialized. You can use the setPersistent function
                    to indicate the contrary and the getPersistent() one to get the current state.
                    
                    Example: 
                       if node.getObject("ThisObject").isPersistent() : 
                                print("This DataField will not be serialized")  
                    This may be usefull if you are implementing editting/modelling features
                    using the script and want the changed made to the scene to be saved.          
                """
                return self.realptr.isPersistent()

        def help(self):
                return "DataField("+self.getName()+"::"+self.getValueTypeString()+"): " + self.realptr.getHelp()

