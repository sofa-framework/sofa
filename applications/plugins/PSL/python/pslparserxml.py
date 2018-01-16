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
import os
import xml
import xml.etree.ElementTree as ET

def toText(rootNode):
        print("(NOT WORKING - PSL XML SAVE: "+str(rootNode))
        dprint(rootNode, "")

def dprint(p,spaces):
    '''Utilitary debug function'''
    for n,v in p:
        print(spaces+str(n)+" : "),
        if isinstance(v, str):
            print(v)
        else:
            print(spaces+"{")
            dprint(v, spaces+"    ")
            print(spaces+"}")

def toAst(xmlnode):
    '''Takes an XMLNode and convert it into the AST structured used by PSL Engine.'''
    childList = []
    for k in xmlnode.attrib:
        v = xmlnode.attrib[k]
        if len(v) > 2 and v[0] == "p" and v[1] == "'" and v[-1] == "'":
            childList.append( (k, ('p', v[2:-1] ) ) )
        else:
            childList.append( (k, ('s', v ) ) )
    childList.reverse()

    for child in xmlnode:
        for j in toAst(child):
            childList.append( j )

    if len(childList) == 0:
        if xmlnode.text == None:
            return [(xmlnode.tag, [])]
        else:
            return [(xmlnode.tag, xmlnode.text)]

    return [(xmlnode.tag, childList)]


def parse(xmlcontent):
    '''Takes a string containing an XML scene and convert it into the AST structured used by PSL Engine.'''
    xmlroot = ET.fromstring(xmlcontent)
    return toAst(xmlroot)
