#!/usr/bin/python
# -*- coding: utf-8 -*-
#/******************************************************************************
#*       SOFA, Simulation Open-Framework Architecture, development version     *
#*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
#*                                                                             *
#* This library is free software; you can redistribute it and/or modify it     *
#* under the terms of the GNU Lesser General Public License as published by    *
#* the Free Software Foundation; either version 2.1 of the License, or (at     *
#* your option) any later version.                                             *
#*                                                                             *
#* This library is distributed in the hope that it will be useful, but WITHOUT *
#* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
#* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
#* for more details.                                                           *
#*                                                                             *
#* You should have received a copy of the GNU Lesser General Public License    *
#* along with this library; if not, write to the Free Software Foundation,     *
#* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
#*******************************************************************************
#*                              SOFA :: Framework                              *
#*                                                                             *
#* Contact information: contact@sofa-framework.org                             *
#******************************************************************************/
#*******************************************************************************
#* Contributors:                                                               *
#*    - damien.marchal@univ-lille1.fr Copyright (C) CNRS                       *
#*                                                                             *
#******************************************************************************/
import Sofa
import pprint
import pslengine
PSLExport = []

def whatis(string, n=10):
    return pslengine.whatis(string,n)

def Node(parent, **kwargs):
    """Create a node and attach it to the provided graph"""
    name = "undefined"
    if kwargs == None:
            kwargs = {}
    elif "name" in kwargs:
        name = kwargs["name"]
        del kwargs["name"]

    child = parent.createChild(name)
    for k in kwargs:
        data = child.getData(k)
        if data != None:
            data.value = kwargs[k]
    return child

class psltemplate(object):
   def __init__(self, f):
       self.function = f
       if not "PSLExport" in f.func_globals:
           f.func_globals["PSLExport"] = []
       f.func_globals["PSLExport"].append(self.function.func_name)

   def __call__(self, node, **kwargs):
       if not "name" in kwargs:
           kwargs["name"] = "undefined"
       templatenode = node.createChild(kwargs["name"])
       self.function(templatenode, **kwargs)
       calledAs = self.function.func_name+"("
       for k in kwargs:
           calledAs += str(k)+"="+repr(kwargs[k])
       calledAs += ")"
       #templatenode.addNewData("psl_instanceof", "PSL", "", "s", "PythonDecoratorTemplate")
       #templatenode.addNewData("psl_templatename", "PSL", "", "s", calledAs)
       return templatenode

for (name,desc) in Sofa.getAvailableComponents():
    code = """def %s(owner, **kwargs):
        \"\"\"%s\"\"\"
        if kwargs == None:
                kwargs = {}
        owner.createObject(\"%s\", **kwargs)
""" % (name,desc,name)
    exec(code)

