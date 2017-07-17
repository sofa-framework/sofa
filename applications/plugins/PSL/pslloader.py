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
#* Contributors: damien.marchal@univ-lille1.fr Copyright (C) CNRS              *
#*                                                                             *
#* Contact information: contact@sofa-framework.org                             *
#******************************************************************************/
import hjson
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

def load(rootNode, filename):
	global sofaRoot
	sofaRoot = rootNode
	filename = os.path.abspath(filename)
	dirname = os.path.dirname(filename) 

	print("PYSCIN LOAD: "+str(filename))
	print("PYSCIN ROOT: "+str(dirname))

	olddirname = os.getcwd()
	os.chdir(dirname)	

	f = open(filename).read()
	r = pslengine.processTree(sofaRoot, "", hjson.loads(f, object_pairs_hook=MyObjectHook()))

	os.chdir(olddirname)	
	return r
