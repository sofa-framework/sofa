import Sofa

import os, os.path
import json
import glob

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
        str_xx += listToStr(x) + " "
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
    elif ext == ".stl":
        return parentNode.createObject('MeshSTLLoader', filename=filename, name=_name, **args)
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

class SceneDataIO:
    """ Read/Write from a scene or sub-scene all the data of each component to/from a json files
        The user gives as input the list of type of component he wants to save the state
        @sa example/sceneDataIO_write.py
        @sa example/sceneDataIO_read.py
    """
    def __init__(self, node=None, classNameList=None):
        # main node which contains the components to update
        self.node = node

        # components name to process
        self.classNameList = classNameList

    """ Read/Write from a scene or sub-scene all the data of each component to/from a json file
        @sa example/sceneDataIO_write.py
        @sa example/sceneDataIO_read.py
    """
    def writeData(self, directory=None):

        # directory where all the data will be stored
        if directory is None:
            directory = "scene_data_at_t_"+str(self.node.getTime())
            try:
                os.makedirs(directory)
            except OSError:
                if not os.path.isdir(directory):
                    raise
        elif not os.path.isdir(directory):
            os.makedirs(directory)

        # lets get all the component of the scene
        visitor = SceneDataIO.SofaVisitor('SceneIOVisitor')
        self.node.executeVisitor(visitor)
        componentList = visitor.componentList

        # process the scene to store each component data
        for component in componentList:
            # we do not treat the components which are not among the accepted components
            if (self.classNameList==None) or (not component.getClassName() in self.classNameList):
                continue
            # if the component is among the accepted class name
            filteredData = dict()
            componentData = component.getDataFields()
            filename = directory + os.sep + component.getContext().name + '_' + component.name +".json"

            for name, value in componentData.iteritems():
                if isinstance(value, (list, dict, str, int, float, bool)):
                    if isinstance(value, (list)) and len(value):
                        filteredData[name] = value
                    elif isinstance(value, (dict)) and len(value):
                        filteredData[name] = value
                    elif isinstance(value, (str)) and (len(value) or value!=""):
                        filteredData[name] = value
                    elif isinstance(value, (int, float, bool)):
                        filteredData[name] = value
                    elif isinstance(value, (unicode)):
                        filteredData[name] = value

            with open(filename,'w') as file:
                try:
                    json.dump(filteredData, file)
                except IOError:
                    raise

        print "[SceneDataIO]: the scene:", os.path.basename(__file__), "data has been save into the directory:", directory
        return 1

    def readData(self, directory=None):

        # Lets check that the directory exists and it is not empty
        if directory == None or not os.path.isdir(directory):
            print "[SceneDataIO]: There is no directory where component data has been stored."
            return -1

        if not len(os.listdir(directory)):
            print "[SceneDataIO]: The selected directory:", directory, "is empty."
            return -1

        nb_json = 0
        for file in os.listdir(directory):
            if file.endswith('.json'):
                nb_json = nb_json +1
        if not nb_json:
            print "[SceneDataIO]: The selected directory:", directory, "do not contains any json files."
            return

        # Lets get all the components of the scene
        visitor = SceneDataIO.SofaVisitor('SceneIOVisitor')
        self.node.executeVisitor(visitor)
        componentList = visitor.componentList

        # process the scene to load each component data
        for component in componentList:
            if (self.classNameList==None) or (not component.getClassName() in self.classNameList):
                continue
            filename = directory + component.getContext().name + '_' + component.name +".json"
            if not os.path.isfile(filename):
                continue
            with open(filename,'r') as file:
                componentData = json.load(file)
                for name, value in componentData.iteritems():
                    if isinstance(value, (list)) and len(value):
                        component.findData(name).value = value
                    elif isinstance(value, (dict)) and len(value):
                        component.findData(name).value = value
                    elif isinstance(value, (str)) and (len(value) or value!=""):
                        component.findData(name).value = value
                    elif isinstance(value, (int, float, bool)):
                        component.findData(name).value = value
                    elif isinstance(value, (unicode)):
                        component.findData(name).value = value.encode("ascii")
            component.reinit()

        print "[SceneDataIO]: the previous scene state has been restored."

        return 1

    """ Internal visitor of the SceneIO component to process each graph component
    """
    class SofaVisitor(object):
        def __init__(self, name):
            self.name = name
            self.componentList = list()

        def processNodeTopDown(self, node):
            self.componentList.extend(node.getObjects())

        def processNodeBottomUp(self, node):
            return

        def treeTraversal(self):
            return
        
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
        
        
def localPath( localfile, filename ):
    ## concatenate the absolute filepath of localfile with filename
    ## returns /abs/path/filename (with /abs/path/localfile)
    return os.path.join(os.path.dirname(os.path.abspath(localfile)), filename)

