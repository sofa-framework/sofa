import math
import os.path
import Sofa
import json

import Sofa

import SofaPython.Tools
from SofaPython.Tools import listToStr as concat
from SofaPython.Tools import listListToStr as lconcat
from SofaPython import Quaternion

printLog = True


def insertLinearMapping(node, dofRigidNode=None, dofAffineNode=None, cell='', assemble=True, geometricStiffness=2, isMechanical=True):
    """ insert the correct Linear(Multi)Mapping
    hopefully the template is deduced automatically by the component
    TODO: better names for input dofRigidNode and dofAffineNode, they can be any kind of nodes
    """
    if dofRigidNode is None and dofAffineNode is None:
        Sofa.msg_error("Flexible.API","insertLinearMapping : no dof given")
    else:
        if dofRigidNode is None:
            return node.createObject(
                "LinearMapping", cell=cell, shapeFunction = 'shapeFunction',
                input="@"+dofAffineNode.getPathName(), output="@.", assemble=assemble, geometricStiffness=geometricStiffness, mapForces=isMechanical, mapConstraints=isMechanical, mapMasses=isMechanical)
        elif dofAffineNode is None:
            return node.createObject(
                "LinearMapping", cell=cell, shapeFunction = 'shapeFunction',
                input="@"+dofRigidNode.getPathName(), output="@.", assemble=assemble, geometricStiffness=geometricStiffness, mapForces=isMechanical, mapConstraints=isMechanical, mapMasses=isMechanical)
        else:
            return node.createObject(
                "LinearMultiMapping", cell=cell, shapeFunction = 'shapeFunction',
                input1="@"+dofRigidNode.getPathName(), input2="@"+dofAffineNode.getPathName(), output="@.", assemble=assemble, geometricStiffness=geometricStiffness, mapForces=isMechanical, mapConstraints=isMechanical, mapMasses=isMechanical)

class Deformable:
    """ This class represents a deformable object built from a mesh.
        Various cases are handled:
        - collision model :
            x static (loaded from file or subset of another static model)
            x linearly mapped from dofs
            x identity mapped from several collision models
            x subset mapped from another collision model
        - visual model :
            x static (loaded from file or subset of another static model)
            x linearly mapped from dof
            x identity mapped from a collision model
            x subset mapped from a collision model
    """
    
    def __init__(self, node, name = None):
        if not name is None :
            self.node=node.createChild(name)
        else:
            self.node=node
        self.name=name
        self.meshLoader=None
        self.topology=None
        self.dofs=None
        self.visual=None
        self.mapping=None
        self.normals=None
        self.mass=None


    def loadMesh(self, meshPath, offset = [0,0,0,0,0,0,1], scale=[1,1,1], triangulate=False):
        r = Quaternion.to_euler(offset[3:])  * 180.0 / math.pi
        self.meshLoader = SofaPython.Tools.meshLoader(self.node, meshPath, translation=concat(offset[:3]) , rotation=concat(r), scale3d=concat(scale), triangulate=triangulate)
        self.topology = self.node.createObject("MeshTopology", name="topology", src="@"+self.meshLoader.name )

    def loadVisual(self, meshPath, offset = [0,0,0,0,0,0,1], scale=[1,1,1], color=[1,1,1,1],**kwargs):
        r = Quaternion.to_euler(offset[3:])  * 180.0 / math.pi
        self.visual =  self.node.createObject("VisualModel", name="model", filename=meshPath, translation=concat(offset[:3]) , rotation=concat(r), scale3d=concat(scale), color=concat(color),**kwargs)
        self.visual.setColor(color[0],color[1],color[2],color[3]) # the previous assignement fails when reloading a scene..
        self.normals = self.visual

    def loadVisualCylinder(self, meshPath, offset = [0,0,0,0,0,0,1], scale=[1,1,1], color=[1,1,1,1],radius=0.01,**kwargs):
        r = Quaternion.to_euler(offset[3:])  * 180.0 / math.pi
        self.visual = self.node.createObject("OglCylinderModel", radius=radius, position="@topology.position", edges="@topology.edges" )
        self.normals = self.visual


    def addVisual(self, color=[1,1,1,1]):
        if self.dofs is None:
            Sofa.msg_error("Flexible.API.Deformable","addVisual : visual mesh not added because there is no dof, use LoadVisual instead to have a static visual mesh ")
        else:   # create a new deformable
            d = Deformable(self.node,"Visual")
            d.visualFromDeformable(self,color)
            return d


    def addVisualCylinder(self,radius=0.01, color=[1,1,1,1]):
        if not self.dofs is None:
            d = Deformable(self.node,"Visual")
            d.visual =  d.node.createObject("OglCylinderModel", name="model",radius=radius, color=concat(color))
            return d


    def visualFromDeformable(self, deformable, color=[1,1,1,1]):
        deformable.node.addChild(self.node)
        self.visual = self.node.createObject("VisualModel", name="model", filename="@"+deformable.meshLoader.getPathName()+".filename", color=concat(color))
        self.visual.setColor(color[0],color[1],color[2],color[3]) # the previous assignement fails when reloading a scene..
        self.mapping = self.node.createObject("IdentityMapping", name="mapping", input='@'+deformable.node.getPathName(),output="@.", mapForces=False, mapConstraints=False, mapMasses=False )
        self.normals = self.meshLoader

    def subsetFromDeformable(self, deformable, indices ):
        if not deformable.topology is None:
            topo = SofaPython.Tools.getObjectPath(deformable.topology)
            self.meshLoader = self.node.createObject("MeshSubsetEngine", template = "Vec3", name='MeshSubsetEngine', inputPosition='@'+topo+'.position', inputTriangles='@'+topo+'.triangles', inputQuads='@'+topo+'.quads', indices=concat(indices))
            self.topology = self.node.createObject("MeshTopology", name="topology", src="@"+self.meshLoader.name)
        if not deformable.dofs is None:
            deformable.node.addChild(self.node)
            self.dofs = self.node.createObject("MechanicalObject", template = "Vec3", name="dofs")
            self.mapping = self.node.createObject("SubsetMapping", name='mapping', indices=concat(indices), input='@'+deformable.node.getPathName(),output="@.")

    def fromDeformables(self, deformables=list()):
        args=dict()
        inputs=[]
        i=1
        map = True
        for s in deformables:
            s.node.addChild(self.node)
            args["position"+str(i)]="@"+SofaPython.Tools.getObjectPath(s.topology)+".position"
            args["triangles"+str(i)]="@"+SofaPython.Tools.getObjectPath(s.topology)+".triangles"
            args["quads"+str(i)]="@"+SofaPython.Tools.getObjectPath(s.topology)+".quads"
            inputs.append('@'+s.node.getPathName())
            if s.dofs is None:
                map=False
            i+=1
        self.meshLoader =  self.node.createObject('MergeMeshes', name='MergeMeshes', nbMeshes=len(inputs), **args )
        self.topology = self.node.createObject("MeshTopology", name="topology", src="@"+self.meshLoader.name )
        if map is True:
            self.dofs = self.node.createObject("MechanicalObject", template = "Vec3", name="dofs")
            self.mapping = self.node.createObject("IdentityMultiMapping", name='mapping',input=SofaPython.Tools.listToStr(inputs),output="@.")

    def addMechanicalObject(self):
        if self.meshLoader is None:
            Sofa.msg_error("Flexible.API.Deformable","addMechanicalObject: no loaded mesh for "+ self.name)
            return
        self.dofs = self.node.createObject("MechanicalObject", template = "Vec3", name="dofs", src="@"+self.meshLoader.name)

    def addNormals(self, invert=False):
        if self.topology is None:
            Sofa.msg_error("Flexible.API.Deformable","addNormals : no topology for "+ self.name)
            return
        pos = '@'+self.topology.name+'.position' if self.dofs is None else  '@'+self.dofs.name+'.position'
        self.normals = self.node.createObject("NormalsFromPoints", template='Vec3', name="normalsFromPoints", position=pos, triangles='@'+self.topology.name+'.triangles', quads='@'+self.topology.name+'.quads', invertNormals=invert )

    def addMass(self,totalMass):
        if self.dofs is None:
            Sofa.msg_error("Flexible.API.Deformable","addMass : no dofs for "+ self.name)
            return
        self.mass = self.node.createObject('UniformMass', totalMass=totalMass)

    def addMapping(self, dofRigidNode=None, dofAffineNode=None, labelImage=None, labels=None, useGlobalIndices=False, useIndexLabelPairs=False, assemble=True, isMechanical=True):
        cell = ''
        if not labelImage is None and not labels is None : # use labels to select specific voxels in branching image
            position="@"+self.topology.name+".position" if not self.topology is None else "@"+self.visual.name+".position"
            offsets = self.node.createObject("BranchingCellOffsetsFromPositions", template="BranchingImageUC", name="cell", position =position, image="@"+SofaPython.Tools.getObjectPath(labelImage.branchingImage)+".image", transform="@"+SofaPython.Tools.getObjectPath(labelImage.branchingImage)+".transform", labels=concat(labels), useGlobalIndices=useGlobalIndices, useIndexLabelPairs=useIndexLabelPairs)
            cell = "@"+SofaPython.Tools.getObjectPath(offsets)+".cell"

        self.mapping = insertLinearMapping(self.node, dofRigidNode, dofAffineNode, cell, assemble, isMechanical=isMechanical)


    def addSkinning(self, armatureNode, indices, weights, assemble=True, isMechanical=True):
        """ Add skinning (linear) mapping based on the armature (Rigid3) in armatureNode using
        """
        self.mapping = self.node.createObject("LinearMapping", template="Rigid3,Vec3", name="mapping", input="@"+armatureNode.getPathName(), indices=concat(indices), weights=concat(weights), assemble=assemble, mapForces=isMechanical, mapConstraints=isMechanical, mapMasses=isMechanical)


    def getFilename(self, filenamePrefix=None, directory=""):
        _filename=filenamePrefix if not filenamePrefix is None else self.name
        _filename+=".json"
        _filename=os.path.join(directory, _filename)
        return _filename

    def write(self, filenamePrefix=None, directory=""):
        """ write weights of the linear mapping
        """
        if self.mapping is None:
            return
        if self.mapping.getClassName().find("Linear") == -1:
            return
        filename = self.getFilename(filenamePrefix,directory)
        data = {'indices': self.mapping.indices, 'weights': self.mapping.weights}
        with open(filename, 'w') as f:
            json.dump(data, f)
            if printLog:
                Sofa.msg_info("Flexible.API.Deformable",'Exported Weights to '+filename)

    def read(self, filenamePrefix=None, directory=""):
        """ read weights of the linear mapping
            WARNING: the mapping should already be created
        """
        if self.mapping is None:
            return
        if self.mapping.getClassName().find("Linear") == -1:
            return
        filename = self.getFilename(filenamePrefix,directory)
        if os.path.isfile(filename):
            data = dict()
            with open(filename,'r') as f:
                data.update(json.load(f))
                self.mapping.indices= str(data['indices'])
                self.mapping.weights= str(data['weights'])
                if printLog:
                    Sofa.msg_info("Flexible.API.Deformable",'Imported Weights from '+filename)

class AffineMass:
    def __init__(self, dofAffineNode):
        self.dofAffineNode = dofAffineNode # where the mechanical state is located
        self.mass = None

    def massFromDensityImage(self, dofNode, dofRigidNode, densityImage, lumping='0'):
        node = dofNode.createChild('Mass')
        node.createObject('MechanicalObject', name='massPoints', template='Vec3')
        insertLinearMapping(node, dofRigidNode, self.dofAffineNode, assemble=False)
        massFromDensity = node.createObject('MassFromDensity',  name="MassFromDensity",  template="Affine,"+"Branching"+densityImage.imageType, image="@"+SofaPython.Tools.getObjectPath(densityImage.branchingImage)+".image", transform="@"+SofaPython.Tools.getObjectPath(densityImage.branchingImage)+'.transform', lumping=lumping)
        self.mass = self.dofAffineNode.createObject('AffineMass', name='mass', massMatrix="@"+SofaPython.Tools.getObjectPath(massFromDensity)+".massMatrix")

    def getFilename(self, filenamePrefix=None, directory=""):
        _filename=filenamePrefix if not filenamePrefix is None else "affineMass"
        _filename+=".json"
        _filename=os.path.join(directory, _filename)
        return _filename

    def read(self, filenamePrefix=None, directory=""):
        filename = self.getFilename(filenamePrefix,directory)

        if os.path.isfile(filename):
            data = dict()
            with open(filename,'r') as f:
                data.update(json.load(f))
                self.mass = self.dofAffineNode.createObject('AffineMass', name='mass', massMatrix=data['massMatrix'])
                if printLog:
                    Sofa.msg_info("Flexible.API.AffineMass",'Imported Affine Mass from '+filename)

    def write(self, filenamePrefix=None, directory=""):
        filename = self.getFilename(filenamePrefix,directory)
        data = {'massMatrix': str(self.mass.findData('massMatrix').value).replace('\n',' ')}
        with open(filename, 'w') as f:
            json.dump(data, f)
            if printLog:
                Sofa.msg_info("Flexible.API.AffineMass",'Exported Affine Mass to '+filename)

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
            Sofa.msg_error("Flexible.API.AffineDof","addMechanicalObject : no source")
            return
        self.src = "@"+SofaPython.Tools.getObjectPath(src)+".position"
        self.dof = self.node.createObject("MechanicalObject", template="Affine", name="dofs", position=self.src, **kwargs)

    def getFilename(self, filenamePrefix=None, directory=""):
        _filename=filenamePrefix if not filenamePrefix is None else "affineDof"
        _filename+=".json"
        _filename=os.path.join(directory, _filename)
        return _filename

    def read(self, filenamePrefix=None, directory=""):
        filename = self.getFilename(filenamePrefix,directory)
        if os.path.isfile(filename):
            data = dict()
            with open(filename,'r') as f:
                data.update(json.load(f))
                self.dof = self.node.createObject("MechanicalObject", name="dofs", template=data['template'], position=data['position'], rest_position=data['rest_position'])
                if printLog:
                    Sofa.msg_info("Flexible.API.AffineDof",'Imported Affine Dof from '+filename)

    def write(self, filenamePrefix=None, directory=""):
        if self.dof is None:
            Sofa.msg_error("Flexible.API.AffineDof","write : no dof")
            return
        filename = self.getFilename(filenamePrefix,directory)
        data = {'template':'Affine', 'rest_position': affineDatatostr(self.dof.rest_position), 'position': affineDatatostr(self.dof.position)}
        with open(filename, 'w') as f:
            json.dump(data, f)
            if printLog:
                Sofa.msg_info("Flexible.API.AffineDof",'Exported Affine Dof to '+filename)


class ShapeFunction:
    """ High-level API to manipulate ShapeFunction
    @todo better handle template
    """
    def __init__(self, node):
        self.node = node
        self.shapeFunction=None
   
    def addVoronoi(self, image, position='', cells='', nbRef=8):
        """ Add a Voronoi shape function using path to position  and possibly cells
        """
        if position =='':
            Sofa.msg_error("Flexible.API.ShapeFunction","addVoronoi : no position")
            return
        imagePath = SofaPython.Tools.getObjectPath(image.branchingImage)
        self.shapeFunction = self.node.createObject(
            "VoronoiShapeFunction", template="ShapeFunction,"+"Branching"+image.imageType, 
            name="shapeFunction", cell=cells,
            position=position,
            src="@"+imagePath, method=0, nbRef=nbRef, bias=True)
   
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
            Sofa.msg_error("Flexible.API.ShapeFunction","addExporter : no shapeFunction")
            return
        sfPath = SofaPython.Tools.getObjectPath(self.shapeFunction)
        self.node.createObject(
            "ImageExporter", template="BranchingImageUI", name="exporterIndices", 
            image="@"+sfPath+".indices", transform="@"+sfPath+".transform",
            filename=self.getFilenameIndices(filenamePrefix, directory),
            exportAtBegin=True, printLog=printLog)
        self.node.createObject(
            "ImageExporter", template="BranchingImageR", name="exporterWeights",
            image="@"+sfPath+".weights", transform="@"+sfPath+".transform",
            filename=self.getFilenameWeights(filenamePrefix, directory), exportAtBegin=True, printLog=printLog)
       
    def addContainer(self, filenamePrefix=None, directory=""):
        self.node.createObject(
            "ImageContainer", template="BranchingImageUI", name="containerIndices", 
            filename=self.getFilenameIndices(filenamePrefix, directory), drawBB=False)
        self.node.createObject(
            "ImageContainer", template="BranchingImageR", name="containerWeights",
            filename=self.getFilenameWeights(filenamePrefix, directory), drawBB=False)
        self.shapeFunction = self.node.createObject(
            "ImageShapeFunctionContainer", template="ShapeFunction,BranchingImageUC", name="shapeFunction",
            position='0 0 0', # dummy value to avoid a warning from baseShapeFunction
            transform="@containerWeights.transform",
            weights="@containerWeights.image", indices="@containerIndices.image")



class FEMDof:
    def __init__(self, node):
        self.node = node
        self.dof = None
        self.mesh = None
        self.mapping = None
        self.mass = None
        self.shapeFunction = None

    def addMesh(self, src=None, **kwargs):
        if src is None:
            Sofa.msg_error("Flexible.API.FEMDof","addMesh : no input mesh")
            return
        self.mesh = self.node.createObject("Mesh", name="mesh", src=src, **kwargs)

    def addMechanicalObject(self, position=None, **kwargs):
        if position is None:
            if not self.mesh is None: # use implicit definition from mesh topology
                self.dof = self.node.createObject("MechanicalObject", template="Vec3", name="dofs", **kwargs)
            else:
                Sofa.msg_error("Flexible.API.FEMDof","addMechanicalObject : no input position")
            return
        self.dof = self.node.createObject("MechanicalObject", template="Vec3", name="dofs", position=position, **kwargs)

    def addUniformMass(self, totalMass):
        self.mass=self.node.createObject("UniformMass", template="Vec3", name="mass", totalMass=totalMass)

    def addShapeFunction(self):
        self.shapeFunction=self.node.createObject("BarycentricShapeFunction", name="shapeFunction")

    def getFilename(self, filenamePrefix=None, directory=""):
        _filename=filenamePrefix if not filenamePrefix is None else "FEMDof"
        _filename+=".json"
        _filename=os.path.join(directory, _filename)
        return _filename

    def read(self, filenamePrefix=None, directory=""):
        filename = self.getFilename(filenamePrefix,directory)
        if os.path.isfile(filename):
            data = dict()
            with open(filename,'r') as f:
                data.update(json.load(f))
                # dofs
                self.dof = self.node.createObject("MechanicalObject", name="dofs", template=data['template'], position=(data['position']), rest_position=(data['rest_position']))
                # mesh
                if 'hexahedra' in data and 'tetrahedra' in data:
                    self.mesh = self.node.createObject("Mesh", name="mesh", position=(data['rest_position']), hexahedra=data['hexahedra'],  tetrahedra=data['tetrahedra'] )
                # uniform mass
                if 'totalMass' in data :
                    self.addUniformMass(data['totalMass'])
                if printLog:
                    Sofa.msg_info("Flexible.API.FEMDof",'Imported FEM Dof from '+filename)

    def readMapping(self, filenamePrefix=None, directory=""):
        """ read mapping parameters
            WARNING: the mapping shoud be already created
        """
        if self.mapping is None:
            return
        filename = self.getFilename(filenamePrefix,directory)
        if os.path.isfile(filename):
            data = dict()
            with open(filename,'r') as f:
                data.update(json.load(f))
                if 'mappingType' in data:
                    if data['mappingType'].find("Linear") != -1:
                        self.mapping.indices= str(data['indices'])
                        self.mapping.weights= str(data['weights'])
                    elif data['mappingType'].find("SubsetMultiMapping") != -1:
                        self.mapping.indexPairs= str(data['indexPairs'])
                    elif data['mappingType'].find("SubsetMapping") != -1:
                        self.mapping.indices= str(data['indices'])
                if printLog:
                    Sofa.msg_info("Flexible.API.FEMDof",'Imported FEM Dof mapping from '+filename)

    def write(self, filenamePrefix=None, directory=""):
        if self.dof is None:
            Sofa.msg_error("Flexible.API.FEMDof","write : no dof")
            return
        filename = self.getFilename(filenamePrefix,directory)
        data = {'template':'Vec3', 'rest_position': lconcat(self.dof.rest_position), 'position': lconcat(self.dof.position)}

        # add mapping data if existing
        if not self.mapping is None:
            if self.mapping.getClassName().find("Linear") != -1:
                data['mappingType']=self.mapping.getClassName()
                data['indices']=self.mapping.indices
                data['weights']=self.mapping.weights
            elif self.mapping.getClassName().find("SubsetMultiMapping") != -1:
                data['mappingType']=self.mapping.getClassName()
                data['indexPairs']=lconcat(self.mapping.indexPairs)
            elif self.mapping.getClassName().find("SubsetMapping") != -1:
                data['mappingType']=self.mapping.getClassName()
                data['indices']=lconcat(self.mapping.indices)

        # add some topology data if existing
        if not self.mesh is None:
            # data['edges']=self.mesh.edges
            # data['triangles']=self.mesh.triangles
            # data['quads']=self.mesh.quads
            data['hexahedra']=lconcat(self.mesh.hexahedra)
            data['tetrahedra']=lconcat(self.mesh.tetrahedra)

        # add mass data if existing
        if not self.mass is None:
            data['totalMass']=self.mass.totalMass

        with open(filename, 'w') as f:
            json.dump(data, f)
            if printLog:
                Sofa.msg_info("Flexible.API.FEMDof",'Exported FEM Dof to '+filename)


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
        self.strainMapping = None
        self.relativeStrainMapping = None
        self.forcefield = None
        self.cell = ''

    def addGaussPointSampler(self, shapeFunction, nbPoints, **kwargs):
        shapeFunctionPath = SofaPython.Tools.getObjectPath(shapeFunction.shapeFunction)
        self.sampler = self.node.createObject(
            "ImageGaussPointSampler", template="BranchingImageR,BranchingImageUC", name="sampler",
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

    def addMechanicalObject(self, dofRigidNode=None, dofAffineNode=None, assemble=True, **kwargs):
        if self.sampler is None:
            Sofa.msg_error("Flexible.API.Behavior","addMechanicalObject : no sampler")
            return
        self.dofs = self.node.createObject("MechanicalObject", template="F"+self.type, name="dofs", **kwargs)
        self.mapping = insertLinearMapping(self.node, dofRigidNode, dofAffineNode, self.cell , assemble)
    

    def getFilename(self, filenamePrefix=None, directory=""):
        _filename=filenamePrefix if not filenamePrefix is None else self.name
        _filename+=".json"
        _filename=os.path.join(directory, _filename)
        return _filename

    def read(self, filenamePrefix=None, directory="", **kwargs):
        filename = self.getFilename(filenamePrefix,directory)
        data = dict()
        with open(filename,'r') as f:
            data.update(json.load(f))
            self.sampler = self.node.createObject('GaussPointContainer',name='GPContainer', volumeDim=data['volumeDim'], inputVolume=data['inputVolume'], position=data['position'], **kwargs)
            if not self.labelImage is None and not self.labels is None:
                celloffsets = self.node.createObject("BranchingCellOffsetsFromPositions", template="BranchingImageUC", name="cell", position ="@"+SofaPython.Tools.getObjectPath(self.sampler)+".position", src="@"+SofaPython.Tools.getObjectPath(self.labelImage.branchingImage), labels=concat(self.labels))
                self.cell = "@"+SofaPython.Tools.getObjectPath(celloffsets)+".cell"
            if printLog:
                Sofa.msg_info("Flexible.API.Behavior",'Imported Gauss Points from '+filename)


    def readWeights(self, filenamePrefix=None, directory=""):
        """ read weights of the linear mapping
            WARNING: the mapping shoud be already created
        """
        if self.mapping is None:
            return
        if self.mapping.getClassName().find("Linear") == -1:
            return
        filename = self.getFilename(filenamePrefix,directory)
        if os.path.isfile(filename):
            data = dict()
            with open(filename,'r') as f:
                data.update(json.load(f))
                self.mapping.indices= str(data['indices'])
                self.mapping.weights= str(data['weights'])
                self.mapping.weightGradients= str(data['weightGradients'])
                self.mapping.weightHessians= str(data['weightHessians'])    
                if printLog:
                    Sofa.msg_info("Flexible.API.Behavior",'Imported Weights from '+filename)

    def write(self, filenamePrefix=None, directory=""):
        filename = self.getFilename(filenamePrefix,directory)
        volumeDim = len(self.sampler.volume)/ len(self.sampler.position) if isinstance(self.sampler.volume, list) is True else 1 # when volume is a list (several GPs or order> 1)
        data = {'volumeDim': str(volumeDim), 'inputVolume': SofaPython.Tools.listListToStr(self.sampler.volume), 'position': SofaPython.Tools.listListToStr(self.sampler.position),
                'indices': self.mapping.indices, 'weights': self.mapping.weights,
                'weightGradients': self.mapping.weightGradients, 'weightHessians': self.mapping.weightHessians}
        # @todo: add restShape ?
        with open(filename, 'w') as f:
            json.dump(data, f)
            if printLog:
                Sofa.msg_info("Flexible.API.Behavior",'Exported Gauss Points to '+filename)

    def writeObj(self, filenamePrefix=None, directory=""):
        filename = self.getFilename(filenamePrefix,directory).replace("json","obj")
        with open(filename, 'w') as f:
            f.write(self.name+"\n")
            for p in self.sampler.position:
                f.write("v "+SofaPython.Tools.listToStr(p)+"\n")
            if printLog:
                Sofa.msg_info("Flexible.API.Behavior",'Exported Gauss Points as a mesh: '+filename)


    def addHooke(self, strainMeasure="Corotational", youngModulus=0, poissonRatio=0, viscosity=0, useOffset=False, assemble=True):
        eNode = self.node.createChild("E")
        eNode.createObject('MechanicalObject',  template="E"+self.type, name="E")
        self.strainMapping = eNode.createObject(strainMeasure+'StrainMapping', template="F"+self.type+",E"+self.type, assemble=assemble)
        if useOffset:
            eOffNode = eNode.createChild("offsetE")
            eOffNode.createObject('MechanicalObject',  template="E"+self.type, name="E")
            self.relativeStrainMapping = eOffNode.createObject('RelativeStrainMapping', template="E"+self.type+",E"+self.type, assemble=assemble)
            self.forcefield = eOffNode.createObject('HookeForceField', name="ff", template="E"+self.type, youngModulus= youngModulus, poissonRatio=poissonRatio, viscosity=viscosity, assemble=assemble, isCompliance=False)
        else:
            self.forcefield = eNode.createObject('HookeForceField', name="ff", template="E"+self.type, youngModulus= youngModulus, poissonRatio=poissonRatio, viscosity=viscosity, assemble=assemble, isCompliance=False)

    def addProjective(self, youngModulus=0, viscosity=0, assemble=True):
        self.forcefield = self.node.createObject('ProjectiveForceField', name="ff", template="F"+self.type,  youngModulus=youngModulus, viscosity=viscosity,assemble=assemble)
