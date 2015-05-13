import Sofa

import os.path
import json

import units

def listToStr(x):
    """ concatenate lists for use with data.
    """
    return ' '.join(map(str, x))

def listListToStr(xx):
    """ concatenate lists of list for use with data.
    """
    str_xx=""
    for x in xx:
        str_xx += listToStr(x)
    return str_xx

def strToListFloat(s):
    """ Convert a string to a list of float
    """
    return map(float,s.split())

def strToListInt(s):
    """ Convert a string to a list of float
    """
    return map(int,s.split())

def getObjectPath(obj):
    """ Return the path of this object
    """
    return obj.getContext().getPathName()+"/"+obj.name

def getNode(rootNode, path):
    """ Return node at path or None if not found
    """
    currentNode = rootNode
    pathComponents = path.split('/')
    for c in pathComponents:
        if len(c)==0: # for leading '/' and in case of '//'
            continue
        currentNode = currentNode.getChild(c)
        if currentNode is None:
            print "SofaPython.Tools.findNode: can't find node at", path
            return None
    return currentNode

def meshLoader(parentNode, filename, name=None, **args):
    """ Insert the correct MeshLoader based on the filename extension
    """
    ext = os.path.splitext(filename)[1]
    if name is None:
        _name="loader_"+os.path.splitext(os.path.basename(filename))[0]
    else:
        _name=name
    if ext == ".obj":
        return parentNode.createObject('MeshObjLoader', filename=filename, name=_name, **args)
    elif ext == ".vtu" or ext == ".vtk":
        return parentNode.createObject('MeshVTKLoader', filename=filename, name=_name, **args)
    else:
        print "ERROR SofaPython.Tools.meshLoader: unknown mesh extension:", ext
        return None

class Material:
    """ This class reads a json file which contains different materials parameters.
        The json file must contain values in SI units
        This class provides an API to access these values, the access methods convert the parameters to the current units, if the requested material or parameter does not exist, the value from the default material is returned
        @sa units.py for units management
        @sa example/material.py for an example
    """

    def __init__(self, filename=None):
        self._reset()
        if not filename is None:
            self.load(filename)

    def _reset(self):
        self.data = dict()
        # to be sure to have a default material
        self.data["default"] = { "density": 1000, "youngModulus": 10e3, "poissonRatio": 0.3 }
        
    def _get(self, material, parameter):
        if material in self.data and parameter in self.data[material]:
            return self.data[material][parameter]
        else:
            return self.data["default"][parameter]
        
    def load(self, filename):
        self._reset()
        with open(filename,'r') as file:
            self.data.update(json.load(file))
            
    def density(self, material):
        return units.massDensity_from_SI(self._get(material, "density"))
    
    def youngModulus(self, material):
        return units.elasticity_from_SI(self._get(material, "youngModulus"))
    
    def poissonRatio(self, material):
        return self._get(material, "poissonRatio")
        
        
class ComponentDataIO:
    """ Read/Write component data to/from a json file
        @sa example/componentDataIO_write.py
        @sa example/componentDataIO_read.py
    """
    def __init__(self, component=None, dataList=[]):
        self.component = component
        self.setDataList(dataList)
        
    def setDataList(self, dataList):
        # TODO check the data exist in the component
        self.dataList = dataList
        
    def writeData(self, filename=None):
        componentData=dict()
        for d in self.dataList:
            componentData[d] = self.component.findData(d).value
        _filename = filename
        if _filename is None:
            _filename = self.component.name+".json"
        with open(_filename,'w') as file:
            json.dump(componentData, file)
        print "[ComponentDataIO]: component:", self.component.name, "data written to:", _filename
        
    def readData(self, filename=None):
        _filename = filename
        if _filename is None:
            _filename = self.component.name+".json"
        with open(_filename,'r') as file:
            componentData = json.load(file)
            for d in self.dataList:
                self.component.findData(d).value = componentData[d]
        print "[ComponentDataIO]: component:", self.component.name, "data read from:", _filename
        
        
        

