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
#* Authors: damien.marchal@univ-lille1.fr Copyright (C) CNRS                   *
#*                                                                             *
#* Contact information: contact@sofa-framework.org                             *
#******************************************************************************/
import hjson
import os

sofaComponents = ["APIVersion", "VisualParam", "MechanicalObject", "VisualModel"]
sofaComponentAttribs = { "Node" : {"name" : "undefined" , "gravity" : "3.0 4.0 5.0" },
			 "APIVersion" : {"name" : "undefined" , "printLogs" : False},
			 "VisualParam" : {"name" : "undefined", "printLogs" : False, "listening" : False },
 			 "MechanicalObject" : {"name" : "undefined", "printLogs" : False, "listening" : False, "positions": "0 0 0" }	 
			}

SofaStackFrame = []

	

class SofaObject(object):
	def __init__(self):
		self.params = {}
		
	def __getattr__(self, name):
		return self.params[name]
		
	def __getitem__(self, name):
		return self.params[name]

	def __setitem__(self, name, value):
		self.params[name] = value

def instantiate(o, g):
	#parent, key, kv, stack, frame
	frame = {}
	for key in g:
		frame[key] = g[key]	
	o.stack.append(frame)
	n, a= o.template[0]
	n=processNode(None, n, a, o.stack, frame)
	o.stack.pop(-1)
	return n

class Template(SofaObject):
	def __init__(self, name, properties, template, stack) :
		self.params = {}
		self.params["name"] = name
		self.properties = properties
		self.template = template
		self.stack = stack 

class Node(SofaObject):
	def __init__(self, **kwargs):
		self.params = {}
		self.params["name"] = "undefined"
		self.params["gravity"] = "3.0 4.0 5.0"
		for i in kwargs:
			self.params[i] = kwargs[i]

		self.children = []
		self.objects = []
		self.templates = []		

	def findChild(self, name):
		for child in self.children:
			if child.name == name:
				return child
		return None
		
	def findTemplate(self, name):
		
		for template in self.templates:
			if template.name == name:
				return template
		
		for child in self.children:
			r = child.findTemplate(name)
			if r != None:
				return r
		return None

	def addChild(self, c):
		self.children.append(c)

	def addObject(self, c):
		self.objects.append(c)
	
	def addTemplate(self, c):
		self.templates.append(c)

	def __repr__(self):
		return "Node "+ str(self.params) 

class VisualParam(SofaObject):
	def __init__(self):
		self.params = {}
		self.params["name"] = "undefined"
		self.params["renderTriangles"] = True

	def __repr__(self):
		return "VisualParam "+ str(self.params) 
		
class APIVersion(SofaObject):
	def __init__(self):
		self.params = {}
		self.params["name"] = "undefined"
		self.params["level"] = "00.00"

	def __repr__(self):
		return "APIVersion "+ str(self.params) 


class VisualModel(SofaObject):
	def __init__(self):
		self.params = {}
		self.params["name"] = "undefined"
		self.params["filename"] = "undefined"
		
	def __repr__(self):
		return "VisualModel "+ str(self.params) 

def dumpSofa(s, prefix=""):
	if isinstance(s, Node):
		print(prefix+"<Node name='"+str(s.name)+">'")
		for c in s.objects:
			dumpSofa(c, prefix+"    ")		
		for c in s.children:
			dumpSofa(c, prefix+"    ")		
		print(prefix+"</Node>")
		
	else:
		print(prefix+str(s))

def flattenStackFrame(sf):
	res = {}
	for frame in sf:
		for k in frame:
			res[k] = frame[k
			]
	return res	

def getFromStack(name, stack):
	for frame in reversed(stack):
		for k in frame: 
			if k == name:
				return frame[k]
	return None

def populateFrame(cname, frame, stack):
	fself = getFromStack("self", stack)
	if fself == None:
		return 
	for aname in fself.params:
		frame[aname] = lambda tname : sys.out.write("T NAME") 
					     

def processPython(key, kv, stack, frame):
	#print("Process Python...")
	r = flattenStackFrame(stack)
	exec(kv, r)


def evalPython(key, kv, stack, frame):
	#print("Process Python...")
	r = flattenStackFrame(stack)
	return eval(kv, r)


def processParameter(name, value, stack, frame):
	#print("PP "+name + " stack frame is: "+str(stack)) 
	if isinstance(value, hjson.OrderedDict):
		print("PP.Dict: ")
	else:
		print("PP.Leaf: " + str(value) + " for "+name) 
		if value[0] == 'p' and value[1] == '"':
			value = evalPython(None, value[2:-1], stack, frame) 
		frame[name] = value
		frame["self"][name] = value
		if name == "name":
			frame["self"]["name"] = value  
			#print(" install a lambda on :" + kv +" -> "+ str(frame["self"]))
			frame[value] = frame["self"] 
		
def createObject(name, stack , frame):
	if name == "VisualParam":
		return VisualParam()
	if name == "APIVersion":
		return APIVersion()
	if name == "VisualModel":
		return VisualModel()
	#print("UNKNOW OBJET")		
	
def processObjectDict(obj, dic, stack, frame):
	#print("PD: "+ str(obj))
	for key,value in dic:
		if key == "Python":				
			processPython(key, value, stack, frame)
		else:
			processParameter(key, value, stack ,frame)
								
def processObject(key, kv, stack, frame):
	#print("PO: "+key + " stack frame is: "+str(stack))
	populateFrame(key, frame, stack)
	if isinstance(kv, list):
		frame = {}
		stack.append(frame)
		frame["self"] = obj = createObject(key, stack, frame)
		processObjectDict(obj, kv, stack, frame)	
		stack.pop(-1) 	
		return obj
	else:
		print("PO.Leaf: ") 
		return None

def processTemplate(parent, key, kv, stack, frame):
	name = "undefined"
	properties = {}
	pattern = [] 
	for key,value in kv:
		if key == "name":
			name = value
		elif key == "properties":
			properties = value
		else:
			pattern.append( (key, value) ) 

	parent.addTemplate( Template(name, properties, pattern, stack) )
	
def processNode(parent, key, kv, stack, frame):
	#print("PN:"+key + " stack frame is: "+str(stack))
	stack.append(frame)
	populateFrame(key, frame, stack)
	tself = frame["self"] = Node()
	if parent != None:
		parent.addChild(tself)
	if isinstance(kv, list):
		for key,value in kv:
			if key == "Node":
				n = processNode(tself, key, value, stack, {})	
			elif key == "Python":	
				processPython(key, value, stack, {})
			elif key == "Template":
				processTemplate(tself, key, value, stack, {})
			elif key in sofaComponents:
				o = processObject(key, value, stack, {})
				tself.addObject(o)
			else:
				processParameter(key, value, stack, frame)
	else:
		print("LEAF: "+kv)
	stack.pop(-1)
	return tself

def processTree(key, kv):
	stack = []
	frame = {}
	if isinstance(kv, list):
		for key,value in kv:
			if key == "Import":
				print("Importing: "+value+".pyjson")
			elif key == "Node":
				processNode(None, key, value, stack, globals())
			elif key == "Python":	
				processPython(key, value, stack, globals())
			elif key in sofaComponents:
				processObject(key, value, stack, globals())
			else:
				processParameter(key, value, stack, frame)
	else:
		print("LEAF: "+kv)


s = open("test1.pyson").read()

class MyObjectHook(object):
	def __call__(self, s):
		return s

d = hjson.loads(s, object_pairs_hook=MyObjectHook())
#print(repr( d ))

processTree("", d) 
