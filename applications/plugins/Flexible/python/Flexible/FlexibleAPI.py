import math
import os.path

import SofaPython.Tools
from SofaPython.Tools import listToStr as concat
from SofaPython import Quaternion

class Deformable:
    
    def __init__(self, node, name):
        self.node = node.createChild( name )
        self.dofs=None
        
    def setMesh(self, position, meshPath):
        r = Quaternion.to_euler(position[3:])  * 180.0 / math.pi
        self.meshLoader = SofaPython.Tools.meshLoader(self.node, meshPath, translation=concat(position[:3]) , rotation=concat(r))
        self.topology = self.node.createObject('MeshTopology', name='topology', src="@"+self.meshLoader.name )
        self.dofs = self.node.createObject("MechanicalObject", template = "Vec3d", name="dofs", src="@"+self.meshLoader.name)
        
    def addVisual(self):
        return Deformable.VisualModel(self.node)
    
    class VisualModel:
        def __init__(self, node ):
            self.node = node.createChild("visual")
            self.model = self.node.createObject('VisualModel', name="model")
            self.mapping = self.node.createObject('IdentityMapping', name="mapping")    

    def addSkinning(self, bonesPath, indices, weights):
        self.skinning = self.node.createObject("LinearMapping", template="Rigid3d,Vec3d", name="skinning", input="@"+bonesPath, indices=concat(indices), weights=concat(weights))

class ShapeFunction:
    def __init__(self, node, name):
        self.node = node.createChild(name)
        self.name = name
        self.shapeFunction=None
   
    def addVoronoi(self, position, image):
        """ Add a Voronoi shape function using position from position component and BranchingImage image
        """
        imagePath = SofaPython.Tools.getObjectPath(image.branchingImage)
        self.shapeFunction = self.node.createObject('VoronoiShapeFunction', name="shapeFunction", template="ShapeFunctiond,BranchingImageD", position="@"+SofaPython.Tools.getObjectPath(position)+".position", src="@"+imagePath, method=0, nbRef=8, bias=True)
   
    def getFilenameIndices(self, filenamePrefix=None, directory=""):
        _filename=filenamePrefix if not filenamePrefix is None else self.name
        _filename+="_indices.mhd"
        _filename=os.path.join(directory, _filename)
        return _filename

    def getFilenameWeights(self, filenamePrefix=None, directory=""):
        _filename=filenamePrefix if not filenamePrefix is None else self.name
        _filename+="_weights.mhd"
        _filename=os.path.join(directory, _filename)
        return _filename

    def addExporter(self, filenamePrefix=None, directory=""):
        if self.shapeFunction is None:
            print "[FelexibleAPI.ShapeFunction] ERROR: no shapeFunction"
        sfPath = SofaPython.Tools.getObjectPath(self.shapeFunction)
        self.node.createObject('ImageExporter', template="BranchingImageUI", name="exporterIndices", image="@"+sfPath+".indices", transform="@"+sfPath+".transform", filename=self.getFilenameIndices(filenamePrefix, directory), exportAtEnd=True, printLog=True)
        self.node.createObject('ImageExporter', template="BranchingImageD", name="exporterWeights", image="@"+sfPath+".weights", transform="@"+sfPath+".transform", filename=self.getFilenameWeights(filenamePrefix, directory), exportAtEnd=True, printLog=True)
       
       
       