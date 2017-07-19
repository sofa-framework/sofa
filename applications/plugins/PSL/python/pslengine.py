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
import difflib
import os

# TODO(dmarchal 2017-06-17) Get rid of these ugly globals.
templates = {}
sofaComponents = []
SofaStackFrame = []
sofaRoot = None
imports = {}

def refreshComponentListFromFactory():
    global sofaComponents
    sofaComponents = []
    for (name, desc) in Sofa.getAvailableComponents():
            sofaComponents.append(name)

def srange(b, e):
        s=""
        for i in range(b,e):
                s+=str(i)+" "
        return s

def flattenStackFrame(sf):
        """Return the stack frame content into a single "flat" dictionnary.
           The most recent entries are overriden the oldest.
           """
        res = {}
        for frame in sf:
                for k in frame:
                        res[k] = frame[k]
        return res

def getFromStack(name, stack):
        """Search in the stack for a given name. The search is proceeding from
           the most recent entries to the oldest. If 'name' cannot be found
           in the stack None is returned. """
        for frame in reversed(stack):
                for k in frame:
                        if k == name:
                                return frame[k]
        return None

def populateFrame(cname, frame, stack):

        """Initialize a frame from the current attributes of the 'self' object
           This is needed to expose the data as first class object.
        """
        fself = getFromStack("self", stack)
        if fself == None:
                return
        #for datafield in fself.getDataFields():
        #	frame[datafield] = lambda tname : sys.out.write("T NAME")


def processPython(parent, key, kv, stack, frame):
        """Process a python fragment of code with context provided by the content of the stack."""
        r = flattenStackFrame(stack)
        l = {}
        exec(kv, r, l)
        ## Apply the local change in the global context
        for k in l:
                stack[-1][k] = l[k]

def evalPython(key, kv, stack, frame):
        """Process a python fragment of code with context provided by the content of the stack."""
        r = flattenStackFrame(stack)
        retval = eval(kv, r)
        return retval

def processParameter(parent, name, value, stack, frame):
        if isinstance(value, list):
                matches = difflib.get_close_matches(name, sofaComponents+templates.keys(), n=4)
                c=parent.createChild("[XX"+name+"XX]")
                Sofa.msg_error(c, "Unknow parameter or Component [" + name + "] suggestions -> "+str(matches))
        else:
                ## Python Hook to build an eval function.
                if value[0] == 'p' and value[1] == '"':
                        value = evalPython(None, value[2:-1], stack, frame)

                try:
                        frame["self"].findData(name).setValueString(str(value))
                except Exception,e:
                        Sofa.msg_error(parent, "Unable to get the argument " + name)

                if name == "name":
                        frame[value] = frame["self"]
                        frame["name"] = value

def createObject(parentNode, name, stack , frame, kv):
        #print("CREATE OBJECT {"+name+"} WITH: "+str(kv)+ "in "+parentNode.name+" stack is:"+str(stack))
        if name in sofaComponents:
                n=None
                if "name" in frame:
                        n = parentNode.createObject(name, **kv)
                else:
                        n = parentNode.createObject(name, **kv)
                return n

        failureObject = parentNode.createObject("Undefined", **kv)
        Sofa.msg_error(failureObject, "Unable to create object "+str(key))
        return failureObject

def processObjectDict(obj, dic, stack, frame):
        for key,value in dic:
                if key == "Python":
                        processPython(obj, key, value, stack, frame)
                else:
                        processParameter(obj, key, value, stack ,frame)

def processObject(parent, key, kv, stack, frame):
    try:
        global sofaComponents
        populateFrame(key, frame, stack)
        frame = {}
        kwargs = {}
        if not isinstance(kv, list):
                kv = [("name" , kv)]

        for k,v in kv:
                if v[0] == 'p' and v[1] == '"':
                        v = evalPython(None, v[2:-1], stack, frame)

                if k == "name":
                        frame["name"] = v

                kwargs[k] = str(v)

        stack.append(frame)
        frame["self"] = obj = createObject(parent, key, stack, frame, kwargs)
        if not "name" in kwargs:
            obj.findData("name").unset()

        stack.pop(-1)

        if key == "RequiredPlugin" :
                refreshComponentListFromFactory()

        return obj
    except Exception:
        c=parent.createChild("[XX"+key+"XX]")
        Sofa.msg_error(c, "Problem in creating the Object")
        return None

# TODO add a warning to indicate that a template is loaded twice.
def importTemplates(content):
        templates = {}
        for key, value in content:
                if key == "Template":
                        name = "undefined"
                        properties = {}
                        rvalue = []
                        for k,v in value:
                                if k == "name":
                                        name = str(v)
                                elif k == "properties":
                                        properties = v
                                else:
                                        rvalue.append((k, v))
                        templates[name] = {"properties":properties, "content" : rvalue}
                else:
                        Sofa.msg_warning("SceneLoaderPYSON", "An imported file contains something that is not a Template.")

        return templates

# TODO gérer les imports circulaires...
def processImport(parent, key, kv, stack, frame):
        global imports, templates
        if not (isinstance(kv, str) or isinstance(kv, unicode)):
                print("Expecting a single 'string' entry....in procesImport " + str(type(kv)))
                return
        filename = kv+".pyson"
        if not os.path.exists(filename):
                dircontent = os.listdir(os.getcwd())
                matches = difflib.get_close_matches(filename, dircontent, n=4)
                Sofa.msg_error(parent, "The file '" + filename + "' does not exists. Do you mean: "+str(matches))
                return
        Sofa.msg_info(parent, "Importing "+ os.getcwd() + "/"+filename)

        f = open(filename).read()
        loadedcontent = hjson.loads(f, object_pairs_hook=MyObjectHook())
        imports[filename] = importTemplates(loadedcontent)
        #print("IMPORTED TEMPLATE: " + str(imports[filename].keys()))

        for tname in imports[filename].keys():
                templates[kv+"."+tname] = imports[filename][tname]
        #print("TEMPLATES: "+str(templates))

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
        o.listening = True
        o.setTemplate(kv)
        frame[str(name)] = o
        templates[str(name)] = o
        return o

aliases = {}
def processAlias(parent, key, kv, stack, frame):
        global aliases
        oldName, newName = kv.split('-')
        aliases[newName]=oldName

def reinstanciateTemplate(templateInstance):
        global templates

        key = templateInstance.name
        frame = {}
        frame["parent"]=templateInstance
        frame["self"]=templateInstance
        nframe = {}
        instanceProperties = eval(templateInstance.src)
        #print("RE-Instanciate template: "+ templateInstance.name )
        #print("             properties: "+ str(instanceProperties) )

        #print("TODO: "+str(dir(templateInstance)))
        for c in templateInstance.getChildren():
                templateInstance.removeChild(c)

        c = templateInstance.getObjects()
        for o in c:
                templateInstance.removeObject(o)

        # Is there a template with this name, if this is the case
        # Retrieve the associated templates .
        if isinstance(templates[key], Sofa.Template):
                #print("SOFA TEMPLATE")
                n = templates[key].getTemplate()
                for k,v in n:
                        if k == 'name':
                                None
                        elif k == 'properties':
                                for kk,vv in v:
                                        if not kk in frame:
                                                nframe[kk] = vv
                        else:
                                source = v
        else:
                source = templates[key]["content"]
                for k,v in templates[key]["properties"]:
                        if not k in frame:
                                nframe[k] = str(v)
        #print("Template: "+str(source))
        #print("Instance properties: "+str(instanceProperties))

        for k,v in instanceProperties:
                nframe[k] = templateInstance.findData(str(k)).getValue(0)

        stack = [globals(), frame]
        n = processNode(templateInstance, "Node", source, stack, nframe, doCreate=False)
        #n.name = key


def instanciateTemplate(parent, key, kv, stack, frame):
        global templates
        print("Instanciate template: "+key + "-> "+str(kv))
        stack.append(frame)
        nframe={}
        source = None
        if isinstance(templates[key], Sofa.Template):
                n = templates[key].getTemplate()
                for k,v in n:
                        if k == 'name':
                                None
                        elif k == 'properties':
                                for kk,vv in v:
                                        if not kk in frame:
                                                nframe[kk] = vv
                        else:
                                source = v
        else:
                source = templates[key]["content"]
                for k,v in templates[key]["properties"]:
                        if not k in frame:
                                nframe[k] = str(v)
        #print("Template: "+str(source))


        for k,v in kv:
                nframe[k] = v
        #print("STACK FRAME IS : " +str(nframe))
        n = processNode(parent, "Node", source, stack, nframe, doCreate=True)
        n.name = key

        if isinstance(templates[key], Sofa.Template):

                for k,v in kv:
                        if not hasattr(n, k):
                                print("ADDING NEW ATTRIBUTE "+str(k)+" -> "+str(v))
                                if isinstance(v, int):
                                        n.addData(k, key+".Properties", "Help", "d", v)
                                elif isinstance(v, str) or isinstance(v,unicode):
                                        n.addData(k, key+".Properties", "Help", "s", str(v))
                                elif isinstance(v, float):
                                        n.addData(k, key+".Properties", "Help", "f", v)
                                elif isinstance(v, unicode):
                                        n.addData(k, key+".Properties", "Help", "f", str(v))
                                #else:
                                #	n.addData(k, key+".Properties", "Help", "s", str(v))
                                #data = n.findData(k)
                                #templates[key].trackData(data)

                n.addData("src", key+".Properties", "No help", "s", repr(kv))
        stack.pop(-1)

def processNode(parent, key, kv, stack, frame, doCreate=True):
        global templates, aliases
        #print("PN:"+ parent.name + " : " + key )
        stack.append(frame)
        populateFrame(key, frame, stack)
        if doCreate:
                tself = frame["self"] = parent.createChild("undefined")
        else:
                tself = frame["self"] = parent
        if isinstance(kv, list):
                for key,value in kv:
                        if isinstance(key, unicode):
                                key = str(key)

                        if key in aliases:
                                #print("Alias resolution to: "+aliases[key])
                                key = aliases[key]

                        if key == "Import":
                                n = processImport(tself, key, value, stack, {})
                        elif key == "Node":
                                n = processNode(tself, key, value, stack, {})
                        elif key == "Python":
                                processPython(tself, key, value, stack, {})
                        elif key == "Template":
                                tself.addObject( processTemplate(tself, key, value, stack, {}) )
                        elif key == "Using":
                                processAlias(tself, key,value, stack, frame)
                        elif key in sofaComponents:
                                o = processObject(tself, key, value, stack, {})
                                if o != None:
                                        tself.addObject(o)
                        elif key in templates:
                                instanciateTemplate(tself, key,value, stack, frame)
                        else:
                                # we are on a cache hit...so we refresh the list.
                                refreshComponentListFromFactory()

                                if key in sofaComponents:
                                        o = processObject(tself, key, value, stack, {})
                                        if o != None:
                                                tself.addObject(o)

                                processParameter(tself, key, value, stack, frame)
        else:
                print("LEAF: "+kv)
        stack.pop(-1)
        return tself

def processTreePSL1(parent, key, kv):
    stack = []
    frame = {}
    if isinstance(kv, list):
            for key,value in kv:
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

def processTree(parent, key, kv, directives):
        if directives["version"] == "1.0":
            return processTreePSL1(parent, key, kv)
        # Add here the future version of the language

