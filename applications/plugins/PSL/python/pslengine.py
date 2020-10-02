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

def instanciate(parent, templateName, **kwargs):
    params = []
    for k in kwargs:
        p = ""
        if isinstance(kwargs[k], list):
            for i in kwargs[k]:
                p = p +  " " + str(i)
        else:
            p = str(kwargs[k])
        params.append((k, p))
    instanciateTemplate(parent, templateName, params, [], {})

pslprefix = "PSL::engine: "
templates = {}
aliases = {}
sofaAliases = {}
sofaComponents = []
sofaHelp = {}
SofaStackFrame = []
datafieldQuirks = []
sofaRoot = None
imports = {}

def refreshComponentListFromFactory():
    global sofaComponents, sofaAliases
    sofaComponents = []
    sofaAliases = {}
    for (name, desc) in Sofa.getAvailableComponents():
            sofaComponents.append(name)
            sofaHelp[name] = desc
            for alias in Sofa.getAliasesFor(name):
                sofaAliases[alias] = name

def srange(b, e):
        s=""
        for i in range(b,e):
                s+=str(i)+" "
        return s

def flattenStackFrame(sf):
        """Return the stack frame content into a single "flat" dictionnary.
           The most recent entries are overwriting the oldest.
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

def findStackLevelFor(key, stack):
    for frame in reversed(stack):
        if key in frame:
            return frame
    return None

def isAStringToken(token, values=("s","p","m")):
    if not isinstance(token, tuple):
        return False
    if not len(token) == 2:
        return False
    if not token[0] in values:
        return False
    return True

def evalPython(key, kv, stack, frame):
        """Process a python fragment of code with context provided by the content of the stack."""
        r = flattenStackFrame(stack)
        retval = eval(kv, r)
        return retval

def getField(object, name):
    d = object.getLink(name)
    if d != None:
        return d

    d = object.getData(name)
    if d != None:
        return d

    return None


def processStringOld(object, name, value, stack, frame):
    ## Python Hook to build an eval function.
    if value[0] == 'p':
        d = object.addNewData("psl_"+name, "PSL", "This hold a python expression.", "s", str(value[1]))
        object.findData("psl_"+name).setPersistant(True)
        return evalPython(None, value[1], stack, frame)
    elif value[0] == 's':
        return value
    else:
        raise Exception("Invlid string token")

def processPython(parent, key, token, stack, frame):
        """Process a python fragment of code with context provided by the content of the stack."""
        if not isAStringToken(token, ("s", "m")):
            Sofa.msg_error(parent, "Python expect string only argument instead of '"+str(token)+"'")
            return None

        #if not isAStringToken(token, ("s", "m")):
        #    raise Exception("Only string or multiline string allowed: "+str(token))

        ## retrieve the value as a string.
        stringvalue = token[1]

        p=parent.createObject("Python", name='"'+stringvalue[0:20]+'..."')
        p.addNewData("psl_source","PSL", "This hold a python expression.", "s", str(stringvalue))

        context = flattenStackFrame(stack)
        local = {}
        exec(stringvalue, context, local)

        ## Transfer the local entries to the previously defined context or in the existing variable
        ## somewhere in the stack frame.
        ## This allow import in one line and use the imported in the following ones.
        for k in local:
                lframe = findStackLevelFor(k, stack)
                if lframe != None:
                    lframe[k] = local[k]
                else:
                    stack[-1][k] = local[k]

def processString(value, stack, frame):
    if not isinstance(value, tuple) or len(value) != 2:
        raise Exception("Invalid value to process: "+str(value))

    ## Python Hook to build a single fragment
    if value[0] == 'p':
        return evalPython(None, value[1], stack, frame)
    ## Python Hook to evaluate a multiline fragment
    elif value[0] == 'm':
        return evalPython(None, value[1], stack, frame)
    ## This is an explicitely given real string.
    elif value[0] == 's':
        return str(value[1])


    raise Exception("Invalid token "+str(value)+ " to process.")


def processParameter(parent, name, value, stack, frame):
        try:
            if isinstance(value, list):
                    matches = difflib.get_close_matches(name, sofaComponents+templates.keys()+sofaAliases.keys(), n=4)
                    c=parent.createChild("[XX"+name+"XX]")
                    Sofa.msg_error(c, pslprefix+" unknow parameter or component [" + name + "] suggestions -> "+str(matches))
            elif not name in datafieldQuirks:
                    ## Python Hook to build an eval function.
                    value = processString(value, stack, frame)

                    try:
                            field = getField(frame["self"], name)
                            if field != None:
                                field.setValueString(str(value))
                                field.setPersistant(True)
                            else:
                                Sofa.msg_error(parent, pslprefix+" unable to get the field '" +name+"'")

                    except Exception,e:
                            Sofa.msg_error(parent, pslprefix+" exception while parsing field '" +name+"' because "+str(e))

                    if name == "name":
                            frame[value] = frame["self"]
                            #frame["name"] = value

        except Exception, e:
            SofaPython.sendMessageFromException(e)
            Sofa.msg_error(parent, pslprefix+" unable to parse parameter '"+str(name)+ "=" + str(value)+"'")

def createObject(parentNode, name, stack , frame, kv):
        if name in sofaComponents:
                obj = parentNode.createObject(name, **kv)
                for k in kv:
                    if getField(obj, k) == None:
                        Sofa.msg_info(obj, pslprefix+" attribute '"+str(k)+"' is a parsing hook. Let's add Data field to fix it. To remove this warning stop using parsing hook.")
                        d = obj.addNewData(k, "PSL", "", "s", str(kv[k]))
                        obj.findData(k).setPersistant(True)
                return obj

        kv["name"] = name
        failureObject = parentNode.createObject("Undefined", **kv)

        Sofa.msg_error(failureObject, pslprefix+" unable to create the object '"+str(name)+"'")
        return failureObject

def processObjectDict(obj, dic, stack, frame):
        for key,value in dic:
                if key == "Python":
                        processPython(obj, key, value, stack, frame)
                else:
                        processParameter(obj, key, value, stack ,frame)

def processObject(parent, key, kv, stack, frame):
        global sofaComponents
        populateFrame(key, frame, stack)
        frame = {}
        kwargs = {}
        if not isinstance(kv, list):
                kv = [("name" , kv)]

        properties = None
        for k,v in kv:
                if k == "properties":
                    if properties == None:
                        properties = v
                    else:
                        c=parent.createChild("[XX"+key+"XX]")
                        Sofa.msg_error(c, pslprefix+" Unable to create an object '"+key+"' because of duplicated properties keywords.")
                        return None
                elif isAStringToken(v, ('s', 'p')):
                    v = processString(v, stack, frame)
                    kwargs[k] = v
                elif isinstance(v, int):
                    kwargs[k] = v
                elif isinstance(v, float):
                        kwargs[k] = v
                else:
                    c=parent.createChild("[XX"+key+"XX]")
                    Sofa.msg_error(c, pslprefix+" Unable to create an object '"+key+"' because of invalid parameter "+str(k)+"="+str(v))
                    return None

        stack.append(frame)
        frame["self"] = obj = createObject(parent, key, stack, frame, kwargs)

        if properties:
            processProperties(obj, obj.name, properties, stack, frame)

        ### Force all the data field into a non-persistant state.
        for datafield in obj.getListOfDataFields():
            datafield.setPersistant(False)

        for link in obj.getListOfLinks():
            link.setPersistant(False)

        ### Then revert only the ones that have been touched
        for dataname in kwargs:
            try:
                if dataname in datafieldQuirks:
                    continue

                field = getField(obj, dataname)
                if field != None:
                    field.setPersistant(True)

            except Exception,e:
                Sofa.msg_warning(obj, pslprefix+" this does not seems to be a valid field '"+str(dataname)+"'")

        if not "name" in kwargs:
            obj.findData("name").unset()

        stack.pop(-1)

        if key == "RequiredPlugin" :
                refreshComponentListFromFactory()

        return obj

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
                                    if isAStringToken(v, ('s')):
                                        name = processString(v, None, None)
                                    else:
                                        Sofa.msg_warning(pslprefix, " Template names must be provided.")
                        templates[name] = value
                else:
                        Sofa.msg_warning(pslprefix, " an imported file contains something that is not a Template.")

        return templates

def getFileToImportFromPartialName(parent, partialname):
        if partialname.endswith(".psl"):
            return partialname

        if partialname.endswith(".pslx"):
            return partialname

        if partialname.endswith(".py"):
            return partialname

        if os.path.exists(partialname+".psl"):
            if os.path.exists(partialname+".py"):
                Sofa.msg_warning(parent, pslprefix+"Both '"+partialname+"'.psl and "+partialname+".py. Importing the psl version.")
            if os.path.exists(partialname+".py"):
                    Sofa.msg_warning(parent, pslprefix+"Both '"+partialname+"'.psl and "+partialname+".pslx. Importing the psl version.")
            return partialname+".psl"

        if os.path.exists(partialname+".pslx"):
            if os.path.exists(partialname+".py"):
                    Sofa.msg_warning(parent, pslprefix+"Both '"+partialname+"'.pslx and "+partialname+".py. Importing the psl version.")
            return partialname+".pslx"

        if os.path.exists(partialname+".py"):
            return partialname+".py"

        return None

def processImportPSL(parent, importname, filename, key, stack, frame):
    global imports, templates

    im = parent.createObject("Import")
    im.name = importname

    f = open(filename).read()
    loadedcontent = pslparserhjson.parse(f)
    imports[filename] = importTemplates(loadedcontent)

    msg = "From <i>"+filename+"</i> importing: <ul>"
    for tname in imports[filename].keys():
            templates[importname+"."+tname] = imports[filename][tname]
            msg+=" <li> template <i>"+tname+"</i> </li>"
    msg += "</ul>"
    Sofa.msg_info(im, msg)


def processImportPSLX(parent, importname, filename, key, stack, frame):
    Sofa.msg_error(parent, pslprefix+"Importing .pslx file format is not supported yet... "+ os.getcwd() + "/"+filename)

def processImportPY(parent, importname, filename, key, stack, frame):
    Sofa.msg_info(parent, pslprefix+"Importing '"+importname+".py' file format.")
    pythonfile = open(filename).read()

    locals = {}
    frame = flattenStackFrame(stack)
    exec(pythonfile, frame, locals)

    if not "PSLExport" in locals:
            Sofa.msg_error(parent, pslprefix+"The file '"+filename+"' does not seem to contain PSL templates.")
            return

    for name in locals:
            templates[importname+"."+name] = locals[name]


# TODO gérer les imports circulaires...
def processImport(parent, key, kv, stack, frame):
        """ This function "import" the file provided as parameter """

        ## Check that the kv value is in fact a string or an unicode
        if not isAStringToken(kv):
                Sofa.msg_error(parent, pslprefix+" to much parameter given in procesImport " + str(type(kv)))
                return

        importname=processString(kv, stack, frame)
        if isinstance(importname, unicode):
            importname=str(importname)

        ## If it is we get complete the filename and try to open it.
        filename = getFileToImportFromPartialName(parent, importname)
        if filename == None or not os.path.exists(filename):
                dircontent = os.listdir(os.getcwd())
                matches = difflib.get_close_matches(filename, dircontent, n=4)
                Sofa.msg_error(parent, pslprefix+" the file '" + filename + "' does not exists. Do you mean: "+str(matches))
                return

        if filename.endswith(".psl"):
            return processImportPSL(parent, importname, filename, key, stack, frame)
        elif filename.endswith(".pslx"):
            return processImportPSLX(parent, importname, filename, key, stack, frame)
        elif filename.endswith(".py"):
            return processImportPY(parent, importname, filename, key, stack, frame)

        Sofa.msg_error(parent, pslprefix+"invalid file format to import "+ os.getcwd() + "/"+filename+ " supproted format are [psl, pslx, py]")

def processTemplate(parent, key, kv, stack, frame):
        global templates
        name = "undefined"
        properties = {}
        pattern = []
        for key,value in kv:
                if key == "name" and isAStringToken(value, ('s')):
                        name = value[1]
                elif key == "properties":
                        properties = value
                else:
                        pattern.append( (key, value) )
        o = parent.createObject("Template", name=str(name))
        o.listening = True
        o.setTemplate(kv)
        o.trackData(o.findData("psl_source"))
        o.addNewData("psl_instanceof", "PSL", "", "s", "Template")
        frame[str(name)] = o
        templates[str(name)] = o
        return o

def processAlias(parent, key, token, stack, frame):
        if not isAStringToken(token, ('p')):
            Sofa.msg_error(parent, pslprefix+" Unable to set an alias because of invalid syntax.")
        global aliases
        value = token[1]
        oldName, newName = value.split(' as ')
        aliases[newName]=oldName

def processProperties(self, key, kv, stack, frame):
    if not isinstance(kv, list):
        raise Exception("This shouldn't happen, exepecting only list")

    msg = ""
    for k,v in kv:
        ## Check if the property named by "k" already exists.
        if hasattr(self, k):
            msg += " - cannot add a property named '"+k+"' as it already exists"
            continue

        if isAStringToken(v):
            v=processString(v, stack, frame)

        if isinstance(v, int):
            self.addNewData(k, "Properties", "", "d", v)
        elif isinstance(v, str) or isinstance(v,unicode):
            self.addNewData(k, "Properties", "", "s", str(v))
        elif isinstance(v, float):
            self.addNewData(k, "Properties", "", "f", v)

        if hasattr(self, k):
            msg += " - adding: '"+str(k)+"' = "+str(v)
        else:
            msg += " - unable to create a property from the value '"+str(v)+"'"

    Sofa.msg_info(self, pslprefix+"Adding a user properties: \n"+msg)

def instanciateTemplate(parent, key, kv, stack, frame):
        global templates
        nframe={}
        parentstack = stack
        stack = [nframe]
        source = None
        properties=[]
        if isinstance(templates[key], Sofa.Template):
                templatesource = templates[key].getTemplate()
        else:
                templatesource = templates[key]

        if isinstance(templatesource, types.FunctionType):
            kwargs = {}
            for k,v in kv:
                if isAStringToken(v):
                    v = processString(v, stack,frame)
                kwargs[k] = v

            n = parent.createChild(key)
            templatesource(n, **kwargs)
        elif isinstance(templatesource, psl.dsl.psltemplate):
            kwargs = {}
            for k,v in kv:
                if isAStringToken(v):
                    v = processString(v, stack,frame)
                kwargs[k] = v
            n=templatesource(parent, **kwargs)
        else:
            ## NOW PROCESSING THE TEMPLATE
            source = []
            for k,v in templatesource:
                    if k == 'name':
                        None
                    elif k == 'properties':
                            for kk,vv in v:
                                    if isAStringToken(vv, ('s','p','m')):
                                        vv=processString(vv, parentstack, frame)

                                    if not kk in frame:
                                            nframe[kk] = vv

                                    properties.append(kk)
                    else:
                            source.append((k,v))

            nframe["args"] = []
            for k,v in kv:
                    if isAStringToken(v, ('s','p','m')):
                        v=processString(v, parentstack, frame)

                    if k in nframe:
                        nframe[k] = v
                    else:
                        nframe["args"].append((k,v))

            if len(source)==1 and source[0][0]=="Node":
                n = processNode(parent, "Node", source[0][1], stack, nframe, doCreate=True)
            else:
                n = processNode(parent, "", source, stack, nframe, doCreate=False)

        ## Add the meta-type information.
        n.addNewData("psl_properties", "PSL", "Captured variables for template re-instantiation", "s", repr(kv))
        n.addNewData("psl_instanceof", "PSL", "Type of the object", "s", str(key))

        stack.pop(-1)

def processNode(parent, key, kv, stack, frame, doCreate=True):
        global templates, aliases
        stack.append(frame)
        populateFrame(key, frame, stack)

        if doCreate:
                if parent == None:
                    tself = Sofa.createNode("undefined")
                else:
                    tself = parent.createChild("undefined")

                frame["self"] = tself

                ### Force all the data field into a non-persistant state.
                for datafield in tself.getListOfDataFields():
                    datafield.setPersistant(False)

                for link in tself.getListOfLinks():
                    link.setPersistant(False)
        else:
                tself = frame["self"] = parent

        try:
            if isinstance(kv, list):
                for key,value in kv:
                        sofaAliasInitialName = None
                        if isinstance(key, unicode):
                                key = str(key)

                        if key in sofaAliases:
                                sofaAliasInitialName = key
                                key = sofaAliases[key]

                        if key in aliases:
                                key = aliases[key]

                        if key == "Import":
                                n = processImport(tself, key, value, stack, {})
                        elif key == "Node":
                                n = processNode(tself, key, value, stack, {})
                        elif key == "Python":
                                processPython(tself, key, value, stack, {})
                        elif key == "properties":
                                processProperties(tself, key, value, stack, {})
                        elif key == "Template":
                                tself.addObject( processTemplate(tself, key, value, stack, {}) )
                        elif key == "Using":
                                processAlias(tself, key,value, stack, frame)
                        elif key in sofaComponents:
                                o = processObject(tself, key, value, stack, {})
                                if o != None:
                                        if isinstance(sofaAliasInitialName, str):
                                            Sofa.msg_warning(o, pslprefix+"'"+key+" was created using the hard coded alias '"+str(sofaAliasInitialName)+"'"+".  \nUsing hard coded aliases is a confusing practice and we advise you to use scene specific alias with the Alias keyword.")
                        elif key in templates:
                                instanciateTemplate(tself, key,value, stack, frame)
                        else:
                                ## we are on a cache hit...so we refresh the list.
                                refreshComponentListFromFactory()

                                if key in sofaComponents:
                                        o = processObject(tself, key, value, stack, {})
                                        if o != None:
                                                tself.addObject(o)
                                processParameter(tself, key, value, stack, frame)
            else:
                raise Exception("This shouldn't happen, expecting only list")
        except Exception,e:
            s=SofaPython.getSofaFormattedStringFromException(e)
            Sofa.msg_error(tself, "Problem while loading file.  <br>"+s)
        stack.pop(-1)
        return tself

def processRootNode(kv, stack, frame):
        global templates, aliases
        stack.append(frame)
        populateFrame("", frame, stack)

        if isinstance(kv, list):
                for key,value in kv:
                        if isinstance(key, unicode):
                                key = str(key)

                        if key == "Node":
                                n = processNode(None, key, value, stack, {})
                                return n
                        else:
                                Sofa.msg_error(tself, pslprefix+"Unable to find a root Node in this file")
                                return None

        Sofa.msg_error(tself, pslprefix+"Unable to find a root Node in this file")
        return None

## Root function that process an abstract tree.
def processTree(ast, directives, globalenv):
        refreshComponentListFromFactory()

        if directives["version"] == "1.0":
            r = processRootNode(ast, [], globalenv)
            return r

        ## Add here the future version of the language

