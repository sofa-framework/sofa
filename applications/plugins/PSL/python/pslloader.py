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
from __future__ import with_statement
import contextlib

import hjson
import Sofa
import os
import pslast
import pslengine
import pslparserxml
import pslparserhjson
import pslparserpickled

@contextlib.contextmanager
def SetPath(newpath):
    '''This context manager is setting & restoring the current path.
       This is very practical to insure that all the processing is done
       where the file is located.
    '''
    curdir= os.getcwd()
    os.chdir(newpath)
    try: yield
    finally: os.chdir(curdir)

def save(rootNode, filename):
        '''Save provided sofa scene into the given filename.
           The file extension indicate the format to follow among
           which "psl, pslx, pslp".
           psl -> hjson
           pslx -> xml
           pslp -> pickled python
        '''
        filename = os.path.abspath(filename)
        dirname = os.path.dirname(filename)

        with SetPath(dirname):
            os.chdir(dirname)
            file = open(filename, "w")
            if filename.endswith(".psl"):
                file.write(pslparserhjson.toText(rootNode))
            elif filename.endswith(".pslx"):
                file.write(pslparserxml.toText(rootNode))
            elif filename.endswith(".pslp"):
                file.write(pslparserpickled.toText(rootNode))
            file.close()

def preProcess(ast):
    '''This function is crawling through the AST to find
       directives that trigger action before actually creating the
       Sofa object.
    '''
    version = None

    ## Check in the ast for specific directives.
    for cmd, value in ast:
        if cmd == "PSLVersion":
            if version == None:
                if pslengine.isAStringToken(value, ('s')):
                    version = pslengine.processString(value, None, None)
                else:
                    raise Exception("PSLVersion must be a string. Found: "+str(value))
            else:
                raise Exception("There is two PSLVersion directive in the same file.")

    if version == None:
        version = "1.0"

    return {"version": version}

def loadAst(filename):
    ast=[]
    filename = os.path.abspath(filename)
    dirname = os.path.dirname(filename)
    with SetPath(dirname):
        os.chdir(dirname)
        if not os.path.exists(filename):
            Sofa.msg_error("PSL", "Unable to open file '"+filename+"'")
            return ast

        f = open(filename).read()
        if filename.endswith(".psl"):
            ast = pslparserhjson.parse(f)
        elif filename.endswith(".pslx"):
            ast = pslparserxml.parse(f)
        elif filename.endswith(".pslp"):
            ast = pslparserpickled.parse(f)
    return ast

def load(filename):
        '''Load the scene contained in the file with the provided name.
           The root node is returned
           Currently the supported formats are psl, pslx, pslp.
           '''
        filename = os.path.abspath(filename)
        dirname = os.path.dirname(filename)
        with SetPath(dirname):
            f = open(filename).read()
            ast=[]
            if filename.endswith(".psl"):
                ast = pslparserhjson.parse(f)
            elif filename.endswith(".pslx"):
                ast = pslparserxml.parse(f)
            elif filename.endswith(".pslp"):
                ast = pslparserpickled.parse(f)

            if len(ast) == 0:
                rootNode = Sofa.createNode("root")
                Sofa.msg_error(rootNode, "The file '"+filename+"' does not contains valid PSL content.")
                return rootNode

            directives = preProcess(ast[0][1])

            if not directives["version"] in ["1.0"]:
                rootNode = Sofa.createNode("root")
                Sofa.msg_error(rootNode, "Unsupported PSLVersion"+str(directives["version"]))
                return rootNode

            g=globals()
            g["__file__"]=filename
            ret = pslengine.processTree(ast, directives, g)
            return ret

        return None
