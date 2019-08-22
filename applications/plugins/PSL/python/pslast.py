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
from pslengine import isAStringToken

def objectToAst(object):
    '''Convert a Sofa object into a PSL AST for further processing.'''
    childList = []

    ### Add a list of child to this Node.
    for datafield in object.getListOfDataFields():
        if datafield.isPersistant():
            if datafield.hasParent():
                childList.append((datafield.name, datafield.getParentPath()))
            else:
                childList.append((datafield.name, datafield.getValueString()))

    return (object.getName(), childList)


def nodeToAst(node):
    '''Convert recursively a Sofa node into a PSL AST for further processing.'''
    childList = []

    ### Add a list of child to this Node.
    for datafield in node.getListOfDataFields():
        if datafield.isPersistant():
            if datafield.hasParent():
                childList.append((datafield.name, datafield.getParentPath()))
            else:
                childList.append((datafield.name, datafield.getValueString()))

    for object in node.getObjects():
        o, param  = objectToAst(object)
        childList.append( (o, param)  )

    for child in node.getChildren():
        c,l = nodeToAst(child)
        childList.append( (c, l) )

    return ("Node", childList)


def cmpf(a, b):
    '''compare to entries in the AST'''
    if a[0] == b[0]:
        return 0

    if a[0][0].isupper():
        return 1

    return  cmp(a[0], b[0])

def reorderAttributes(ast1):
    '''Reorder attributes in a specific way to facilitate AST comparison.
       This function is just there to help in implementing validity testing
       by comparing ast.
    '''
    if isinstance(ast1, unicode):
        ast1 = str(ast1)

    if isinstance(ast1, str):
        return ast1

    if isinstance(ast1, tuple):
        return (reorderAttributes(ast1[0]), reorderAttributes(ast1[1]))

    if isinstance(ast1, list):
        res = []
        for sub in ast1:
            res.append( reorderAttributes(sub) )

        if len(res) != 0 and isinstance(res[0], tuple):
            res=sorted(res, cmp=cmpf)

        return res

    raise Exception("Trying to convert an invalid AST "+str(type(ast1)))


def removeUnicode(ast1):
    '''Recursively replace unicode with string)'''
    if isinstance(ast1, unicode):
        ast1 = str(ast1)

    if isinstance(ast1, str):
        return ast1

    if isinstance(ast1, tuple):
        return (removeUnicode(ast1[0]), removeUnicode(ast1[1]))

    if isinstance(ast1, list):
        res = []
        for sub in ast1:
            res.append( removeUnicode(sub) )
        return res

    raise Exception("Trying to convert an invalid AST "+str(type(ast1)))

def compareAst(ast1, ast2):
    '''Compare two AST. Returns (True, "") or (False, "Explaination") '''
    if len(ast1) != len(ast2):
        return (False, "Ast mismatch at: \n"+pprint.pformat(ast1)+"\n"+pprint.pformat(ast2))

    if type(ast1) != type(ast2):
        return (False, "Ast type mismatch for: \n"+pprint.pformat(ast1)+"\n"+pprint.pformat(ast2))

    if isinstance(ast1, str):
        if ast1 != ast2:
            return (False, "Ast mismatch at: \n"+pprint.pformat(ast1)+"\n"+pprint.pformat(ast2))
        return (True, "")

    if isAStringToken(ast1):
        if not isAStringToken(ast2):
            return (False, "Ast mismatch at: \n"+pprint.pformat(ast1)+"\n"+pprint.pformat(ast2))
        if ast1[0] != ast2[0] or ast1[1] != ast2[1]:
            return (False, "Ast mismatch at: \n"+pprint.pformat(ast1)+"\n"+pprint.pformat(ast2))
        return (True, "")

    if isAStringToken(ast2):
        if not isAStringToken(ast1):
            return (False, "Ast mismatch at: \n"+pprint.pformat(ast1)+"\n"+pprint.pformat(ast2))
        if ast1[0] != ast2[0] or ast1[1] != ast2[1]:
            return (False, "Ast mismatch at: \n"+pprint.pformat(ast1)+"\n"+pprint.pformat(ast2))
        return (True, "")


    for i in range(0, len(ast1)):
        if ast1[i][0] != ast2[i][0]:
            return (False, "Ast mismatch at \n"+pprint.pformat(ast1)+"\n"+pprint.pformat(ast2))

        s, m = compareAst(ast1[i][1], ast2[i][1])
        if not s:
            return (False, "Ast mismatch at \n"+pprint.pformat(ast1)+"\n"+pprint.pformat(ast2))

    return (True, "")

def sceneToAst(rootNode):
    '''Convert the provided Sofa scene into an AST.'''
    return [nodeToAst(rootNode)]
