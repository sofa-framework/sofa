import math
import os.path
import Sofa
import json

import Flexible.IO
import sys

import SofaPython.Tools
from SofaPython.Tools import listToStr as concat
from SofaPython import Quaternion

def insertLinearMapping(node, dofRigidNode=None, dofAffineNode=None, position=None, cell='', assemble=True, geometricStiffness=2):
    """ insert the correct Linear(Multi)Mapping
    hopefully the template is deduced automatically by the component
    """
    if dofRigidNode is None and dofAffineNode is None:
        print "[Flexible.API.insertLinearMapping] ERROR: no dof given"
    else:
        if dofRigidNode is None:
            return node.createObject(
                "LinearMapping", cell=cell, shapeFunction = 'shapeFunction',
                input="@"+dofAffineNode.getPathName(), output="@.", assemble=assemble)
        elif dofAffineNode is None:
            return node.createObject(
                "LinearMapping", cell=cell, shapeFunction = 'shapeFunction',
                input="@"+dofRigidNode.getPathName(), output="@.", assemble=assemble, geometricStiffness=geometricStiffness)
        else:
            return node.createObject(
                "LinearMultiMapping", cell=cell, shapeFunction = 'shapeFunction',
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

    def addMesh(self, meshPath, offset = [0,0,0,0,0,0,1], scale=[1,1,1], triangulate=False):
        r = Quaternion.to_euler(offset[3:])  * 180.0 / math.pi
        self.meshLoader = SofaPython.Tools.meshLoader(self.node, meshPath, translation=concat(offset[:3]) , rotation=concat(r), scale3d=concat(scale), triangulate=triangulate)
        self.topology = self.node.createObject("MeshTopology", name="topology", src="@"+self.meshLoader.name )
        self.dofs = self.node.createObject("MechanicalObject", template = "Vec3d", name="dofs", src="@"+self.meshLoader.name)

    def addFromDeformables(self, deformables=list()):
        args=dict()
        inputs=[]
        i=1
        for s in deformables:
            s.node.addChild(self.node)
            args["position"+str(i)]="@"+SofaPython.Tools.getObjectPath(s.topology)+".position"
            args["triangles"+str(i)]="@"+SofaPython.Tools.getObjectPath(s.topology)+".triangles"
            args["quads"+str(i)]="@"+SofaPython.Tools.getObjectPath(s.topology)+".quads"
            inputs.append('@'+s.node.getPathName())
            i+=1

        self.meshLoader =  self.node.createObject('MergeMeshes', name='MergeMeshes', nbMeshes=len(inputs), **args )
        self.topology = self.node.createObject("MeshTopology", name="topology", src="@"+self.meshLoader.name )
        self.dofs = self.node.createObject("MechanicalObject", template = "Vec3d", name="dofs")
        self.mapping = self.node.createObject("IdentityMultiMapping", template = "Vec3d,Vec3d",name='mapping',input=SofaPython.Tools.listToStr(inputs),output="@.")

    def addNormals(self, invert=False):
        self.normals = self.node.createObject("NormalsFromPoints", template='Vec3d', name="normalsFromPoints", position='@'+self.dofs.name+'.position', triangles='@'+self.topology.name+'.triangles', quads='@'+self.topology.name+'.quads', invertNormals=invert )

    def addMass(self,totalMass):
        self.mass = self.node.createObject('UniformMass', totalMass=totalMass)

    def addMapping(self, dofRigidNode=None, dofAffineNode=None, labelImage=None, labels=None, assemble=True):
        cell = ''
        if not labelImage is None and not labels is None : # use labels to select specific voxels in branching image
           offsets = self.node.createObject("BranchingCellOffsetsFromPositions", template="BranchingImageUC", name="cell", position ="@"+SofaPython.Tools.getObjectPath(self.topology)+".position", src="@"+SofaPython.Tools.getObjectPath(labelImage.branchingImage), labels=concat(labels))
           cell = "@"+SofaPython.Tools.getObjectPath(offsets)+".cell"
        self.mapping = insertLinearMapping(self.node, dofRigidNode, dofAffineNode, self.topology, cell, assemble)

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
        self.mass = None

    def massFromDensityImage(self, dofRigidNode, densityImage, lumping='0'):
        node = self.node.createChild('Mass')
        dof = node.createObject('MechanicalObject', name='massPoints', template='Vec3d')
        insertLinearMapping(node, dofRigidNode, self.dofAffineNode, dof, assemble=False)
        densityImage.addBranchingToImage('0') # MassFromDensity on branching images does not exist yet
        massFromDensity = node.createObject('MassFromDensity',  name="MassFromDensity",  template="Affine,ImageD", image="@"+SofaPython.Tools.getObjectPath(densityImage.image)+".image", transform="@"+SofaPython.Tools.getObjectPath(densityImage.image)+'.transform', lumping=lumping)
        self.mass = self.dofAffineNode.createObject('AffineMass', name='mass', massMatrix="@"+SofaPython.Tools.getObjectPath(massFromDensity)+".massMatrix")

    def getFilename(self, filenamePrefix=None, directory=""):
        _filename=filenamePrefix if not filenamePrefix is None else "affineMass"
        _filename+=".json"
        _filename=os.path.join(directory, _filename)
        return _filename

    def read(self, filenamePrefix=None, directory=""):
        filename = self.getFilename(filenamePrefix,directory)
        data = dict()
        with open(filename,'r') as f:
            data.update(json.load(f))
        self.mass = self.dofAffineNode.createObject('AffineMass', name='mass', massMatrix=data['massMatrix'])
        print 'Imported Affine Mass from '+filename

    def write(self, filenamePrefix=None, directory=""):
        filename = self.getFilename(filenamePrefix,directory)
        data = {'massMatrix': str(self.mass.findData('massMatrix').value).replace('\n',' ')}
        with open(filename, 'w') as f:
            json.dump(data, f)
        print 'Exported Affine Mass to '+filename

# fix of Sofa<->python serialization
def affineDatatostr(data):
        L = ""
        for it in data :
                for i in xrange(3):
                        L = L+ str(it[i])+" "
                L = L+ "["
                for i in xrange(9):
                        L = L+ str(it[3+i])+" "
                L = L+ "] "
        return L

class AffineDof:
    def __init__(self, node):
        self.node = node
        self.dof = None
        self.src = ''   # source of initial node positions

    def addMechanicalObject(self, src, **kwargs):
        if src is None:
            print "[Flexible.API.AffineDof] ERROR: no source"
        self.src = "@"+SofaPython.Tools.getObjectPath(src)+".position"
        self.dof = self.node.createObject("MechanicalObject", template="Affine", name="dofs", position=self.src, **kwargs)

    def getFilename(self, filenamePrefix=None, directory=""):
        _filename=filenamePrefix if not filenamePrefix is None else "affineDof"
        _filename+=".json"
        _filename=os.path.join(directory, _filename)
        return _filename

    def read(self, filenamePrefix=None, directory=""):
        filename = self.getFilename(filenamePrefix,directory)
        data = dict()
        with open(filename,'r') as f:
            data.update(json.load(f))
        self.dof = self.node.createObject("MechanicalObject", template="Affine", name="dofs", position=data['position'], rest_position=data['rest_position'])
        print 'Imported Affine Dof from '+filename

    def write(self, filenamePrefix=None, directory=""):
        filename = self.getFilename(filenamePrefix,directory)
        data = {'template':'Affine', 'rest_position': affineDatatostr(self.dof.rest_position), 'position': affineDatatostr(self.dof.position)}
        with open(filename, 'w') as f:
            json.dump(data, f)
        print 'Exported Affine Dof to '+filename


class ShapeFunction:
    """ High-level API to manipulate ShapeFunction
    @todo better handle template
    """
    def __init__(self, node):
        self.node = node
        self.shapeFunction=None
   
    def addVoronoi(self, image, position=None, cells=''):
        """ Add a Voronoi shape function using position from position component and BranchingImage image
        """
        if position is None:
            print "[Flexible.API.ShapeFunction] ERROR: no position"
        imagePath = SofaPython.Tools.getObjectPath(image.branchingImage)
        self.shapeFunction = self.node.createObject(
            "VoronoiShapeFunction", template="ShapeFunctiond,"+"Branching"+image.imageType, 
            name="shapeFunction", cells=cells,
            position="@"+SofaPython.Tools.getObjectPath(position)+".position",
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
            exportAtBegin=True, printLog=True)
        self.node.createObject(
            "ImageExporter", template="BranchingImageD", name="exporterWeights", 
            image="@"+sfPath+".weights", transform="@"+sfPath+".transform",
            filename=self.getFilenameWeights(filenamePrefix, directory), exportAtBegin=True, printLog=True)
       
    def addContainer(self, filenamePrefix=None, directory=""):
        self.node.createObject(
            "ImageContainer", template="BranchingImageUI", name="containerIndices", 
            filename=self.getFilenameIndices(filenamePrefix, directory), drawBB=False)
        self.node.createObject(
            "ImageContainer", template="BranchingImageD", name="containerWeights", 
            filename=self.getFilenameWeights(filenamePrefix, directory), drawBB=False)
        self.shapeFunction = self.node.createObject(
            "ImageShapeFunctionContainer", template="ShapeFunctiond,BranchingImageUC", name="shapeFunction",
            position='0 0 0', # dummy value to avoid a warning from baseShapeFunction
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
        self.cell = ''

    def addGaussPointSampler(self, shapeFunction, nbPoints, **kwargs):
        shapeFunctionPath = SofaPython.Tools.getObjectPath(shapeFunction.shapeFunction)
        self.sampler = self.node.createObject(
            "ImageGaussPointSampler", template="BranchingImageD,BranchingImageUC", name="sampler",
            indices="@"+shapeFunctionPath+".indices", weights="@"+shapeFunctionPath+".weights", transform="@"+shapeFunctionPath+".transform", 
            method="2", order=self.type[2:], targetNumber=nbPoints,
            mask="@"+SofaPython.Tools.getObjectPath(self.labelImage.branchingImage)+".branchingImage", maskLabels=concat(self.labels),
            **kwargs)
        celloffsets = self.node.createObject("BranchingCellOffsetsFromPositions", template="BranchingImageUC", name="cell", position ="@"+SofaPython.Tools.getObjectPath(self.sampler)+".position", src="@"+SofaPython.Tools.getObjectPath(self.labelImage.branchingImage), labels=concat(self.labels))
        self.cell = "@"+SofaPython.Tools.getObjectPath(celloffsets)+".cell"

    def addTopologyGaussPointSampler(self, mesh, order="2", **kwargs):
        meshPath = SofaPython.Tools.getObjectPath(mesh)
        self.sampler = self.node.createObject("TopologyGaussPointSampler", name="sampler", method="0", inPosition="@"+meshPath+".position", order=order, **kwargs)
        self.cell = "@"+SofaPython.Tools.getObjectPath(self.sampler)+".cell"

    def addMechanicalObject(self, dofRigidNode=None, dofAffineNode=None, assemble=True):
        if self.sampler is None:
            print "[Flexible.API.Behavior] ERROR: no sampler"
        self.dofs = self.node.createObject("MechanicalObject", template="F"+self.type, name="dofs")
        self.mapping = insertLinearMapping(self.node, dofRigidNode, dofAffineNode, self.sampler, self.cell , assemble)
    

    def getFilename(self, filenamePrefix=None, directory=""):
        _filename=filenamePrefix if not filenamePrefix is None else self.name
        _filename+=".json"
        _filename=os.path.join(directory, _filename)
        return _filename

    def read(self, filenamePrefix=None, directory=""):
        filename = self.getFilename(filenamePrefix,directory)
        data = dict()
        with open(filename,'r') as f:
            data.update(json.load(f))
        self.sampler = self.node.createObject('GaussPointContainer',name='GPContainer', volumeDim=data['volumeDim'], inputVolume=data['inputVolume'], position=data['position'])
        print 'Imported Gauss Points from '+filename

    def write(self, filenamePrefix=None, directory=""):
        filename = self.getFilename(filenamePrefix,directory)
        volumeDim = len(self.sampler.volume)/ len(self.sampler.position) if isinstance(self.sampler.volume, list) is True else 1 # when volume is a list (several GPs or order> 1)
        data = {'volumeDim': str(volumeDim), 'inputVolume': str(self.sampler.volume).replace('[', '').replace("]", '').replace(",", ' '), 'position': str(self.sampler.position).replace('[', '').replace("]", '').replace(",", ' ')}
        with open(filename, 'w') as f:
            json.dump(data, f)
        print 'Exported Gauss Points to '+filename



    def addHooke(self, strainMeasure="Corotational", youngModulus=0, poissonRatio=0, viscosity=0, assemble=True):
        eNode = self.node.createChild("E")
        eNode.createObject('MechanicalObject',  template="E"+self.type, name="E")
        eNode.createObject(strainMeasure+'StrainMapping', template="F"+self.type+",E"+self.type, assemble=assemble)
        eNode.createObject('HookeForceField',  template="E"+self.type, youngModulus= youngModulus, poissonRatio=poissonRatio, viscosity=viscosity, assemble=assemble, isCompliance=False)

    def addProjective(self, youngModulus=0, viscosity=0, assemble=True):
        self.node.createObject('ProjectiveForceField', template="F"+self.type,  youngModulus=youngModulus, viscosity=viscosity,assemble=assemble)

