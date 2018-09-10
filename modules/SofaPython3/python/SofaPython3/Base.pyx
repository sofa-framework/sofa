# -*- coding: ASCII -*-
# This file contains the binding code betwen the sofa 'Base' object and python.

## Imports all the 'C++ types' out of the wrapper
from libcpp.string cimport string as libcpp_string

from .cpp.sofa.core.objectmodel.BaseData_wrap cimport BaseData as _BaseData
from .cpp.sofa.helper.vector_wrap cimport vector as _SofaVector

## Imports the needed 'extension types' out of pyx/pxd files.
from . cimport BaseData

cdef class Base:
    """ Hold a sofa Base object providing access to object's name and datafield
        If other functionnalites are needed you can downcast this object to
        the other corresponding type using the constructors. If the conversion
        is not possible a TypeError exception is risen.

        Example:
          sel_asBase = Editor.getSelected() # return the Base object that is currently selected
          sel_asBaseObject = BaseObject(selection)
    """
    #@staticmethod
    #cdef createFrom(_Base* aBase):
    #            cdef Base py_obj = Base.__new__(Base)
    #            py_obj.realptr = aBase
    #            return py_obj

    def __init__(self, Base src not None):
        self.realptr = src.realptr

    def __getattr__(self, item):
                if item == "__members__":
                        f = self.getDataFieldNames()
                        d = []
                        for ff in f:
                                d.append("d_"+ff+"")
                        return d
                dfname = item
                if item.startswith("d_"):
                        dfname = item[2:]
                r = self.findData(dfname)
                if r != None:
                        return r
                raise AttributeError("has no attribute '"+item+"'")

    def __setattr__(self, item, values):
                dfname = item
                if item.startswith("d_"):
                        dfname = item[2:]
                r = self.findData(dfname)
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

    def findData(self, name):
                """ Searchs and returns for a data field given its name

                    example:
                        obj.findData("template")
                """
                cdef _BaseData* data = self.realptr.findData(name)
                if data==NULL:
                        return None
                return BaseData.toPython(data)

    def getDataFieldNames(self):
                """ Returns a list with the name of all the data field """
                cdef _SofaVector[_BaseData*] v = (self.realptr.getDataFields())
                p = []
                for bd in v:
                        p.append(bd.getName())
                return p

    def getName(self):
                """ Return the name of this object """
                cdef libcpp_string _r = (self.realptr).getName()
                py_result = <libcpp_string>_r
                return py_result

    def getTypeName(self):
        return <libcpp_string>self.realptr.getTypeName()

    def getClassName(self):
        """ Returns the sofa component's name"""
        return <libcpp_string>self.realptr.getClassName()

    def getTemplateName(self):
        return <libcpp_string>self.realptr.getTemplateName()

    def __str__(self):
                return "Base["+self.realptr.getClassName()+"]("+self.realptr.getName()+")"

