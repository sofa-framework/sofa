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
import Sofa
import difflib
import hjson
import os

templates = {}
sofaComponents = []
for (name, desc) in Sofa.getAvailableComponents():
	sofaComponents.append(name)

SofaStackFrame = []

sofaRoot = None
	

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
	def __init__(self, parent, **kwargs):
		self.params = {}
		self.params["name"] = "undefined"
		self.params["gravity"] = "3.0 4.0 5.0"
		for i in kwargs:
			self.params[i] = kwargs[i]

		self.children = []
		self.objects = []
		self.templates = []		
		self.sofaObject = parent.createChild(self.params["name"])
	
	def findData(self, name):
		return self.sofaObject.findData(name)
		
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

def dumpSofa(s, prefix=""):
	res = ""
	if isinstance(s, Node):
		res += prefix+"<Node name='"+str(s.name)+">'"+ "\n"
		for c in s.objects:
			res += dumpSofa(c, prefix+"    ") + "\n"		
		for c in s.children:
			res += dumpSofa(c, prefix+"    ") + "\n"
		res += prefix+"</Node>" + "\n"
	else:
		res += prefix+str(s) + "\n"
	return res
	
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
	for datafield in fself.getDataFields():
		frame[datafield] = lambda tname : sys.out.write("T NAME") 
					     

def processPython(parent, key, kv, stack, frame):
	#print("Process Python...")
	try:
		r = flattenStackFrame(stack)
		exec(kv, r)
	except Exception, e:
		Sofa.msg_error(parent.name, "Unable to process Python "+str(e)+str("\n")+kv)

def evalPython(key, kv, stack, frame):
	#print("Process Python...")
	r = flattenStackFrame(stack)
	return eval(kv, r)


def processParameter(parent, name, value, stack, frame):
	if isinstance(value, list):
		matches = difflib.get_close_matches(name, sofaComponents, n=4)
		Sofa.msg_error(parent.name, "Unknow parameter or Component [" + name + "] suggestions -> "+str(matches))
	else:		
		## Python Hook to build an eval function. 
		if value[0] == 'p' and value[1] == '"':
			value = evalPython(None, value[2:-1], stack, frame) 
				
		try:
			frame["self"].findData(name).setValue(0, str(value))
		except Exception,e:
			Sofa.msg_error(parent.name, "Unable to get the argument " + name)
			
		if name == "name":
			frame[value] = frame["self"] 
			frame["name"] = value
		
def createObject(parentNode, name, stack , frame):
	if name in sofaComponents:
		#print("CREATE OBJET: " + str(frame["name"]))
		n = parentNode.createObject(name, name=str(frame["name"]))
		return n
	return parentNode.createObject("Undefined")
	
def processObjectDict(obj, dic, stack, frame):
	for key,value in dic:
		if key == "Python":				
			processPython(obj, key, value, stack, frame)
		else:
			processParameter(obj, key, value, stack ,frame)
								
def processObject(parent, key, kv, stack, frame):
	populateFrame(key, frame, stack)
	frame = {}
	if not isinstance(kv, list):
		kv = [("name" , kv)]
	frame["name"] = kv[0][1]
	stack.append(frame)
	frame["self"] = obj = createObject(parent, key, stack, frame)
	processObjectDict(obj, kv, stack, frame)	
	stack.pop(-1) 	
	return obj
	
def processTemplate(parent, key, kv, stack, frame):
	global templates 
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
	o = parent.createObject("Template", name=str(name))
	
	o.setTemplate(value)
	frame[str(name)] = o
	templates[str(name)] = o  
	print("SETTEMPLATE: "+str(name)+" -> "+str(o.getTemplate()))
	
	return o
	
def instanciateTemplate(parent, key, kv, stack, frame):
	global templates
	print("Instanciate template: "+key + "-> "+str(kv))
	for k,v in kv:
		frame[k] = v
	n = templates[key].getTemplate()
	print("tempLATE: "+str(n))
	processNode(parent, "Node", n, stack, frame)
	
def processNode(parent, key, kv, stack, frame):
	global templates
	#print("PN:"+ parent.name + " : " + key + " stack frame is: "+str(stack))
	stack.append(frame)
	populateFrame(key, frame, stack)
	tself = frame["self"] = parent.createChild("undefined")
	if isinstance(kv, list):
		for key,value in kv:
			if key == "Node":
				n = processNode(tself, key, value, stack, {})	
			elif key == "Python":	
				processPython(tself, key, value, stack, {})
			elif key == "Template":
				tself.addObject( processTemplate(tself, key, value, stack, {}) )
			elif key in sofaComponents:
				o = processObject(tself, key, value, stack, {})
				tself.addObject(o)
			elif key in templates:
				instanciateTemplate(tself, key,value, stack, frame)
			else:
				processParameter(tself, key, value, stack, frame)
	else:
		print("LEAF: "+kv)
	stack.pop(-1)
	return tself

def processTree(parent, key, kv):
	stack = []
	frame = {}
	if isinstance(kv, list):
		for key,value in kv:
			print("PROCESSING:" + str(key))
			if key == "Import":
				print("Importing: "+value+".pyjson")
			elif key == "Node":
				processNode(parent, key, value, stack, globals())
			elif key == "Python":	
				processPython(parent, key, value, stack, globals())
			elif key in sofaComponents:
				processObject(parent, key, value, stack, globals())
			else:
				processParameter(parent, key, value, stack, frame)
	else:
		print("LEAF: "+kv)



class MyObjectHook(object):
	def __call__(self, s):
		return s

def load(rootNode, filename):
	global sofaRoot 
	sofaRoot = rootNode
	print("PYSCIN LOAD: "+str(filename))
	print("PYSCIN root: "+str(sofaRoot.name))

	f = open(filename).read()
	r = processTree(sofaRoot,"", hjson.loads(f, object_pairs_hook=MyObjectHook()))
	return r
	
	#return "ZOB"
