# -*- coding: UTF8 -*
from libcpp cimport bool
from libcpp.string cimport string as libcpp_string
from base cimport Base as _Base
from basedata cimport BaseData as _BaseData
from sofavector cimport vector as sofavector

cdef class Base:
    """ Hold a sofa Base object providing access to object's name and datafield  
        If other functionnalites are needed you can downcast this object to
        the other corresponding type using the constructors. If the conversion
        is not possible a TypeError exception is risen. 
        
        Example: 
          sel_asBase = Editor.getSelected() # return the Base object that is currently selected  
          sel_asBaseObject = BaseObject(selection)
    """
    cdef _Base* realptr
    
    def findData(self, name):
                """ Searchs and returns for a data field given its name
                        
                    example:
                        obj.findData("template")    
                """
                cdef _BaseData* data = self.realptr.findData(name)
                if data==NULL:
                        return None
                py_data = BaseData()
                py_data.realptr = data 
                return py_data
               
    def getName(self):
                """ Return the name of this object """
                cdef libcpp_string _r = (self.realptr).getName()
                py_result = <libcpp_string>_r
                return py_result 
                
    def __str__(self):
                return "Base["+self.realptr.getClassName()+"]("+self.realptr.getName()+")" 
  
    def getTypeName(self):
        return <libcpp_string>self.realptr.getTypeName() 
    
    def getClassName(self):
        """ Returns the sofa component's name"""
        return <libcpp_string>self.realptr.getClassName()    
        
    def getTemplateName(self):
        return <libcpp_string>self.realptr.getTemplateName()    
        
    def getDataNames(self):
                """ Returns a list with the name of all the data field """
                cdef sofavector[_BaseData*] v = (self.realptr.getDataFields()) 
                p = []
                for bd in v:
                        p.append(bd.getName())
                return p        

    @staticmethod
    cdef createFrom(_Base* aBase):
                cdef Base py_obj = Base.__new__(Base)
                py_obj.realptr = aBase 
                return py_obj 

    def __init__(self, Base src not None):
        self.realptr = src.realptr    

    def __getattr__(self, item):
                r = self.findData(item)
                if r != None:
                        return r 
                raise AttributeError("has no attribute '"+item+"'")

    def __setattr__(self, item, values):
                r = self.findData(item)
                if r:
                        for i in range(0, min(len(r), len(values))):
                                r[i] = values[i]
                else:
                        raise AttributeError("__setattr__()...has no attribute '"+item+"'")

    def __richcmp__(Base self not None,  Base other, op):    
        if op == 2: 
                return self.realptr == other.realptr     
        if op == 3: 
                return self.realptr != other.realptr     
        
        return NotImplemented()
