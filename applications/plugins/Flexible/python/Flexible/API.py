import math
import os.path
import pickle
import Sofa

import Flexible.IO
import sys

import SofaPython.Tools
from SofaPython.Tools import listToStr as concat
from SofaPython import Quaternion

def insertLinearMapping(node, dofRigidNode=None, dofAffineNode=None, position=None, labelImage=None, labels=None, assemble=True, geometricStiffness=2):
    """ insert the correct Linear(Multi)Mapping
    hopefully the template is deduced automatically by the component
    """
    if dofRigidNode is None and dofAffineNode is None:
        print "[Flexible.API.insertLinearMapping] ERROR: no dof given"
    else:
        cell = ''
        if not labelImage is None and not labels is None : # use labels to select specific voxels in branching image
            node.createObject(
                "BranchingCellOffsetsFromPositions", template="BranchingImageUC", name="cell",
                position ="@"+SofaPython.Tools.getObjectPath(position)+".position",
                src="@"+SofaPython.Tools.getObjectPath(labelImage.branchingImage), labels=concat(labels))
            cell = "@cell.cell"

        if dofRigidNode is None:
            return node.createObject(
                "LinearMapping", cell=cell,
                input="@"+dofAffineNode.getPathName(), output="@.", assemble=assemble)
        elif dofAffineNode is None:
            return node.createObject(
                "LinearMapping", cell=cell,
                input="@"+dofRigidNode.getPathName(), output="@.", assemble=assemble, geometricStiffness=geometricStiffness)
        else:
            return node.createObject(
                "LinearMultiMapping", cell=cell,
                input1="@"+dofRigidNode.getPathName(), input2="@"+dofAffineNode.getPathName(), output="@.", assemble=assemble, geometricStiffness=geometricStiffness)

class Deformable:
    """ This class reprents a deformable object build from a mesh.
    @todo: make it multi mesh
    """
    
    def __init__(self, node, name):
        self.node=node.createChild(name)
        self.name=name
        self.dofs=None
        self.meshLoader=None
        self.topology=None
        self.mass=None
        self.visual=None
        self.mapping=None
        self.normals=None
        self.subset=None

    def addMesh(self, meshPath, offset = [0,0,0,0,0,0,1]):
        r = Quaternion.to_euler(offset[3:])  * 180.0 / math.pi
        self.meshLoader = SofaPython.Tools.meshLoader(self.node, meshPath, translation=concat(offset[:3]) , rotation=concat(r))
        self.topology = self.node.createObject("MeshTopology", name="topology", src="@"+self.meshLoader.name )
        self.dofs = self.node.createObject("MechanicalObject", template = "Vec3d", name="dofs", src="@"+self.meshLoader.name)

    def addNormals(self, invert=False):
        self.normals = self.node.createObject("NormalsFromPoints", template='Vec3d', name="normalsFromPoints", position='@'+self.dofs.name+'.position', triangles='@'+self.topology.name+'.triangles', quads='@'+self.topology.name+'.quads', invertNormals=invert )

    def addMass(self,totalMass):
        self.mass = self.node.createObject('UniformMass', totalMass=totalMass)

    def addMapping(self, dofRigidNode=None, dofAffineNode=None, labelImage=None, labels=None, assemble=True):
        self.mapping = insertLinearMapping(self.node, dofRigidNode, dofAffineNode, self.topology, labelImage, labels, assemble)

    def addVisual(self, color=[1,1,1,1]):
        self.visual = Deformable.VisualModel(self.node, color)

    class VisualModel:
        def __init__(self, node, color=[1,1,1,1] ):
            self.node = node.createChild("visual")
            self.model = self.node.createObject("VisualModel", name="model", color=concat(color))
            self.mapping = self.node.createObject("IdentityMapping", name="mapping")

    def addSubset(self, indices ):
        self.subset = Deformable.SubsetModel(self.node, indices, self.topology)

    class SubsetModel:
        def __init__(self, node, indices, topology=None):
            self.node = node.createChild("subset")
            self.dofs = self.node.createObject("MechanicalObject", template = "Vec3d", name="dofs")
            self.mapping = self.node.createObject("SubsetMapping", template = "Vec3d,Vec3d", indices=concat(indices))
            if topology:
                self.subsetEngine = self.node.createObject("MeshSubsetEngine", template = "Vec3d", inputPosition='@../'+topology.name+'.position', inputTriangles='@../'+topology.name+'.triangles', inputQuads='@../'+topology.name+'.quads', indices=concat(indices))
                self.topology = self.node.createObject("MeshTopology", name="topology", src="@"+self.subsetEngine.name )
        def addNormals(self, invert=False):
            self.normals = self.node.createObject("NormalsFromPoints", template='Vec3d', name="normalsFromPoints", position='@'+self.dofs.name+'.position', triangles='@'+self.topology.name+'.triangles', quads='@'+self.topology.name+'.quads', invertNormals=invert )
        def addVisual(self, color=[1,1,1,1]):
            self.visual = Deformable.VisualModel(self.node, color)



# TODO: refactor this
    def addSkinning(self, bonesPath, indices, weights, assemble=True):
        self.mapping = self.node.createObject(
            "LinearMapping", template="Rigid3d,Vec3d", name="skinning",
            input="@"+bonesPath, indices=concat(indices), weights=concat(weights), assemble=assemble)

class AffineMass:
    def __init__(self, node, dofAffineNode):
        self.node = node # a children node of all nodes and shape function
        self.dofAffineNode = dofAffineNode # where the mechanical state is located

    def massFromDensityImage(self, dofRigidNode, densityImage, lumping='0'):
        node = self.node.createChild('Mass')
        dof = node.createObject('MechanicalObject', name='massPoints', template='Vec3d')
        insertLinearMapping(node, dofRigidNode, self.dofAffineNode, dof, assemble=False)
        densityImage.addBranchingToImage('0') # MassFromDensity on branching images does not exist yet
        massFromDensity = node.createObject('MassFromDensity',  name="MassFromDensity",  template="Affine,ImageD", image="@"+SofaPython.Tools.getObjectPath(densityImage.converter)+".image", transform="@"+SofaPython.Tools.getObjectPath(densityImage.converter)+'.transform', lumping=lumping)
        self.dofAffineNode.createObject('AffineMass', name='mass', massMatrix="@"+SofaPython.Tools.getObjectPath(massFromDensity)+".massMatrix")

    def read(self, directory):
#        with open(os.path.join(directory,"affineMass.pkl"), "r") as f:
#            data = pickle.load(f)
#            self.dofAffineNode.createObject('AffineMass', name='mass', massMatrix=data.mass)
#            print 'Imported Affine Mass from '+directory+"/affineMass.pkl"
        sys.path.insert(0, directory)
        __import__('affineMass').loadMass(self.dofAffineNode)
        print 'Imported Affine Mass from '+directory+"/affineMass.py"

    class InternalData:
        def __init__(self,node):
            self.mass = None

    def write(self, directory):
        self.dofAffineNode.createObject('PythonScriptController', filename=__file__, classname='massExporter', variables=directory)

class massExporter(Sofa.PythonScriptController):
    def bwdInitGraph(self,node):
        directory = self.findData('variables').value[0][0]
#        with open(os.path.join(directory,"affineMass.pkl"), "w") as f:
#            pickle.dump(self.InternalData(node), f)
#        print 'Exported Affine Mass in '+directory+"/affineMass.pkl";
        Flexible.IO.export_AffineMass(node.getObject('mass'), directory+"/affineMass.py")
        print 'Exported Affine Mass in '+directory+"/affineMass.py"
        return 0
    class InternalData:
        def __init__(self,node):
            self.mass = str(node.getObject('mass').massMatrix).replace('\n',' ')

class ShapeFunction:
    """ High-level API to manipulate ShapeFunction
    @todo better handle template
    """
    def __init__(self, node, position=None):
        self.node = node
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
        _filename=filenamePrefix if not filenamePrefix is None else "SF"
        _filename+="_indices.mhd"
        _filename=os.path.join(directory, _filename)
        return _filename

    def getFilenameWeights(self, filenamePrefix=None, directory=""):
        _filename=filenamePrefix if not filenamePrefix is None else "SF"
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
    def __init__(self, node, name, type="331", labelImage=None, labels=None):
        self.node = node.createChild(name)
        self.name = name
        self.labelImage = labelImage
        self.labels = labels
        self.type = type
        self.sampler = None
        self.dofs = None
        self.mapping = None

    def addGaussPointSampler(self, shapeFunction, nbPoint):
        shapeFunctionPath = SofaPython.Tools.getObjectPath(shapeFunction.shapeFunction)
        self.sampler = self.node.createObject(
            "ImageGaussPointSampler", template="BranchingImageD,BranchingImageUC", name="sampler",
            indices="@"+shapeFunctionPath+".indices", weights="@"+shapeFunctionPath+".weights", transform="@"+shapeFunctionPath+".transform", 
            method="2", order=self.type[2:], targetNumber=nbPoint,
            mask="@"+SofaPython.Tools.getObjectPath(self.labelImage.branchingImage)+".branchingImage", maskLabels=concat(self.labels), clearData=True)
        
    def addMechanicalObject(self, dofRigidNode=None, dofAffineNode=None, assemble=True):
        if self.sampler is None:
            print "[Flexible.API.Behavior] ERROR: no sampler"
        self.dofs = self.node.createObject("MechanicalObject", template="F"+self.type, name="dofs")
        self.mapping = insertLinearMapping(self.node, dofRigidNode, dofAffineNode, self.sampler, self.labelImage, self.labels, assemble)
    
    def addHooke(self, strainMeasure="Corotational", youngModulus=0, poissonRatio=0, viscosity=0, assemble=True):
        eNode = self.node.createChild("E")
        eNode.createObject('MechanicalObject',  template="E"+self.type, name="E")
        eNode.createObject(strainMeasure+'StrainMapping', template="F"+self.type+",E"+self.type, assemble=assemble)
        eNode.createObject('HookeForceField',  template="E"+self.type, youngModulus= youngModulus, poissonRatio=poissonRatio, viscosity=viscosity, assemble=assemble, isCompliance=False)
        
        
