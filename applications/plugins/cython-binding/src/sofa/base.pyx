# -*- coding: UTF8 -*
from libcpp cimport bool
from libcpp.string cimport string as libcpp_string

from sofa.basedata cimport BaseData, _BaseData
from sofa.sofavector cimport vector as sofavector

cdef class Base(object):
    __dict__ = {}

    """ Hold a sofa Base object providing access to object's name and datafield  
        If other functionnalites are needed you can downcast this object to
        the other corresponding type using the constructors. If the conversion
        is not possible a TypeError exception is risen. 
        
        Example: 
          sel_asBase = Editor.getSelected() # return the Base object that is currently selected  
          sel_asBaseObject = BaseObject(selection)
    """
    def findData(self, name):
                """ Searchs and returns for a data field given its name
                        
                    example:
                        obj.findData("template")    
                """
                cdef _BaseData* data = self.realptr.findData(name)
                
                if data==NULL:
                        return None
                        
                return BaseData.createBaseDataFrom(data)
               
    def getName(self):
                """ Return the name of this object """
                print("HELLO WORLD "+str(<long>self.realptr))
                print("MSG: "+self.realptr.getName())
                
                cdef libcpp_string _r = (self.realptr).getName()
                py_result = <libcpp_string>_r
                return py_result 
                
    def __str__(self):
                return "Base["+self.realptr.getClassName()+"]("+self.realptr.getName()+")" 
  
    def getTypeName(self):
        """ Returns the sofa component's name"""
        return <libcpp_string>self.realptr.getTypeName() 
    
    def getClassName(self):
        """ Returns the sofa component's name"""
        return <libcpp_string>self.realptr.getClassName()    
        
    def getTemplateName(self):
        """ Returns the sofa template name"""
        return <libcpp_string>self.realptr.getTemplateName()    
        
    def getDataNames(self):
                """ Returns a list with the name of all the data field """
                cdef sofavector[_BaseData*] v = (self.realptr.getDataFields()) 
                p = []
                for bd in v:
                        p.append(bd.getName())
                return p        

    @staticmethod
    cdef createBaseFrom(_Base* aBase):
                cdef Base py_obj = Base.__new__(Base)
                super(Base, py_obj).__init__()
                py_obj.realptr = aBase 
                return py_obj 

    def __init__(self, Base src not None):
                super(Base, self).__init__()
                self.realptr = src.realptr    

    #def __dir__(self):
    #            return self.getDataNames() 

    def __getattr__(self, item):
                r = self.findData(item)
                if r != None:
                        return r 
                        
                if item in self.__dict__:
                        return self.__dict__[item]
        
                raise AttributeError("has no attribute '"+item+"'")

    def __setattr__(self, item, values):
                r = self.findData(item)
                if r:
                        r.setValues(values)
                        return
                
                # Fall back to the default python way of adding atributes. 
                self.__dict__[item] = values 
               

    def __richcmp__(Base self not None,  Base other, op):    
        if op == 2: 
                return self.realptr == other.realptr     
        if op == 3: 
                return self.realptr != other.realptr     
        
        return NotImplemented()
