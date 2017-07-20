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
import hjson
import Sofa
import os
import pslengine

class MyObjectHook(object):
        def __call__(self, s):
                return s

def saveTree(rootNode, space):
        print(space+"Node : {")
        nspace=space+"    "
        for child in rootNode.getChildren():
                saveTree(child, nspace)

        for obj in rootNode.getObjects():
                print(nspace+obj.getClassName() + " : { " )
                print(nspace+"    name : "+str(obj.name))
                print(nspace+" } ")

        print(space+"}")

def save(rootNode, filename):
        print("PYSCIN SAVE: "+str(filename))
        saveTree(rootNode,"")

def preProcess(ast):
    version = None
    # Check in the ast for specific directives.
    for cmd, value in ast:
        if cmd == "PSLVersion":
            if version == None:
                if isinstance(value, str) or isinstance(value, unicode):
                    version = str(value)
                else:
                    raise Exception("PSLVersion must be a string in format '1.0'")
            else:
                raise Exception("There is two PSLVersion directive in the same file.")

    if version == None:
        version = "1.0"

    return {"version": version}

def load(rootNode, filename):
        global sofaRoot

        sofaRoot = rootNode
        filename = os.path.abspath(filename)
        dirname = os.path.dirname(filename)

        olddirname = os.getcwd()
        os.chdir(dirname)

        f = open(filename).read()
        ast = hjson.loads(f, object_pairs_hook=MyObjectHook())

        if len(ast) == 0:
            Sofa.msg_error(rootNode, "The file '"+filename+"' does not contains PSL content")
            return rootNode

        directives = preProcess(ast[0][1])

        if not directives["version"] in ["1.0"]:
            Sofa.msg_error(rootNode, "Unsupported PSLVersion"+str(directives["version"]))
            r=rootNode
        else:
            r = pslengine.processTree(sofaRoot, "", ast, directives)

        os.chdir(olddirname)
        return r
