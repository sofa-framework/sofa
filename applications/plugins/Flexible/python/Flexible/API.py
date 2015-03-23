import math
import os.path

import SofaPython.Tools
from SofaPython.Tools import listToStr as concat
from SofaPython import Quaternion

def insertLinearMapping(node, dofRigid=None, dofAffine=None, position=None, labelImage=None, labels=None, assemble=True):
    """ insert the correct Linear(Multi)Mapping
    hopefully the template is deduced automatically by the component
    """
    if dofRigid is None and dofAffine is None:
        print "[Flexible.API.insertLinearMapping] ERROR: no dof given"
    if dofAffine is None:
        #TODO
        return None
    else:
        node.createObject(
            "BranchingCellOffsetsFromPositions", template="BranchingImageUC", name="cell", 
            position ="@"+SofaPython.Tools.getObjectPath(position)+".position", 
            src="@"+SofaPython.Tools.getObjectPath(labelImage.branchingImage), labels=concat(labels))
        if dofRigid is None:
            #TODO
            return None
        else:
            return node.createObject(
                "LinearMultiMapping", cell="@cell.cell", 
                input1="@"+dofRigid.getPathName(), input2="@"+dofAffine.getPathName(), output="@.", assemble=assemble)

class Deformable:
    
    def __init__(self, node, name):
        self.node=node.createChild(name)
        self.name=name
        self.dofs=None
        
    def setMesh(self, position, meshPath):
        r = Quaternion.to_euler(position[3:])  * 180.0 / math.pi
        self.meshLoader = SofaPython.Tools.meshLoader(self.node, meshPath, translation=concat(position[:3]) , rotation=concat(r))
        self.topology = self.node.createObject("MeshTopology", name="topology", src="@"+self.meshLoader.name )
        self.dofs = self.node.createObject("MechanicalObject", template = "Vec3d", name="dofs", src="@"+self.meshLoader.name)
        
    def addVisual(self):
        return Deformable.VisualModel(self.node)
    
    class VisualModel:
        def __init__(self, node ):
            self.node = node.createChild("visual")
            self.model = self.node.createObject("VisualModel", name="model")
            self.mapping = self.node.createObject("IdentityMapping", name="mapping")    

    def addSkinning(self, bonesPath, indices, weights):        
        self.skinning = self.node.createObject(
            "LinearMapping", template="Rigid3d,Vec3d", name="skinning",
            input="@"+bonesPath, indices=concat(indices), weights=concat(weights))

class ShapeFunction:
    """ High-level API to manipulate ShapeFunction
    @todo better handle template
    """
    def __init__(self, node, name, position=None):
        self.node = node.createChild(name)
        self.name = name
        self.position = position # component which contains shape function position (spatial coordinates of the parent nodes)
        self.shapeFunction=None
   
    def addVoronoi(self, image):
        """ Add a Voronoi shape function using position from position component and BranchingImage image
        """
        if self.position is None:
            print "[Flexible.API.ShapeFunction] ERROR: no position"
        imagePath = SofaPython.Tools.getObjectPath(image.branchingImage)
        self.shapeFunction = self.node.createObject(
            "VoronoiShapeFunction", template="ShapeFunctiond,"+"Branching"+image.imageType, 
            name="shapeFunction",
            position="@"+SofaPython.Tools.getObjectPath(self.position)+".position",
            src="@"+imagePath, method=0, nbRef=8, bias=True)
   
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
            print "[Flexible.API.ShapeFunction] ERROR: no shapeFunction"
        sfPath = SofaPython.Tools.getObjectPath(self.shapeFunction)
        self.node.createObject(
            "ImageExporter", template="BranchingImageUI", name="exporterIndices", 
            image="@"+sfPath+".indices", transform="@"+sfPath+".transform",
            filename=self.getFilenameIndices(filenamePrefix, directory),
            exportAtEnd=True, printLog=True)
        self.node.createObject(
            "ImageExporter", template="BranchingImageD", name="exporterWeights", 
            image="@"+sfPath+".weights", transform="@"+sfPath+".transform",
            filename=self.getFilenameWeights(filenamePrefix, directory), exportAtEnd=True, printLog=True)
       
    def addContainer(self, filenamePrefix=None, directory=""):
        if self.position is None:
            print "[Flexible.API.ShapeFunction] ERROR: no position"
        self.node.createObject(
            "ImageContainer", template="BranchingImageUI", name="containerIndices", 
            filename=self.getFilenameIndices(filenamePrefix, directory), drawBB=False)
        self.node.createObject(
            "ImageContainer", template="BranchingImageD", name="containerWeights", 
            filename=self.getFilenameWeights(filenamePrefix, directory), drawBB=False)
        self.shapeFunction = self.node.createObject(
            "ImageShapeFunctionContainer", template="ShapeFunctiond,BranchingImageUC", name="shapeFunction", position="@"+SofaPython.Tools.getObjectPath(self.position)+".position",
            transform="@containerWeights.transform",
            weights="@containerWeights.image", indices="@containerIndices.image")
        
        
class Behavior:
    """ High level API to add a behavior
    """
    def __init__(self, node, name, labelImage=None, labels=None):
        self.node=node.createChild(name)
        self.name=name
        self.labelImage=labelImage
        self.labels=labels
        self.sampler=None
        
    def addGaussPointSampler(self, shapeFunction, nbPoint):
        shapeFunctionPath = SofaPython.Tools.getObjectPath(shapeFunction.shapeFunction)
        self.sampler = self.node.createObject(
            "ImageGaussPointSampler", template="BranchingImageD,BranchingImageUC", name="sampler",
            indices="@"+shapeFunctionPath+".indices", weights="@"+shapeFunctionPath+".weights", transform="@"+shapeFunctionPath+".transform", 
            method="2", order="1", targetNumber=nbPoint, 
            mask="@"+SofaPython.Tools.getObjectPath(self.labelImage.branchingImage)+".branchingImage", maskLabels=concat(self.labels), clearData=True)
        
    def addMechanicalObject(self, dofRigid=None, dofAffine=None):
        if self.sampler is None:
            print "[Flexible.API.Behavior] ERROR: no sampler"
        self.node.createObject("MechanicalObject", template="F331", name="dofs")
        insertLinearMapping(self.node, dofRigid, dofAffine, self.sampler, self.labelImage, self.labels)
    
    def addHooke(self, youngModulus):
        eNode = self.node.createChild("E")
        eNode.createObject('MechanicalObject',  template="E331", name="E")
        eNode.createObject('CorotationalStrainMapping', template="F331,E331", assemble=True)
        eNode.createObject('HookeForceField',  template="E331", youngModulus= SofaPython.units.elasticity_from_SI(youngModulus), poissonRatio=0.4, viscosity=0, assemble=True, isCompliance=False)
        
        
