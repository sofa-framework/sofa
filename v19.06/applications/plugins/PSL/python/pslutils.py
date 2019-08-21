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
import SofaPython
import difflib
import os
import psl.dsl
import pprint
import types
import pslparserhjson

def whatis(name, n=5):
    ret = difflib.get_close_matches(name, sofaComponents+templates.keys()+sofaAliases.keys(), n=n)
    res = "Searching for <i>"+ name + "</i> returns: <br><ul>"
    for i in ret:
        if i in sofaComponents:
            res += "<li>"+i+" (a component) <br> "+sofaHelp[i]+"</li>"
        elif i in templates:
            res += "<li>"+i+" (a template)</li>"
        elif i in templates:
            res += "<li>"+i+" (an alias)</li>"
    res += "</ul>"
    return res
