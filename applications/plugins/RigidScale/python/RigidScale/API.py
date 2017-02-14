#from operator import ge

import sys
import math
import numpy

import Sofa

import Compliant.Frame as Frame
import Compliant.Tools as Tools
from Compliant.Tools import cat as concat


from SofaPython import Quaternion
import SofaPython.Tools
import SofaPython.mass

import SofaPython.Quaternion as quat

import SofaImage.API
import Flexible.Serialization as serialization
import Flexible.API

# to specify the floating point encoding (double by default)
template_suffix=''

# global variable to give a different name to each visual model
idxVisualModel = 0

# to use geometric_stiffness of rigid mappings
# @warning WIP, the API will change
geometric_stiffness = 0

# target scale used when the scale is close to 0
target_scale = [1E-3, 1E-3, 1E-3]

class ShearlessAffineBody:
    
    # Generic Body composed by one or more affine without shear
    # TODO: test/update reading/writing file

    def __init__(self, node, name):
        # node creation
        self.name = name
        self.node = node.createChild(name)
        self.rigidNode = self.node.createChild(name + '_rigid')  # rigid node
        self.scaleNode = self.node.createChild(name + '_scale')  # scale node
        self.affineNode = self.rigidNode.createChild(name + '_affine')  # affine node is a child of both rigid and scale node
        self.scaleNode.addChild(self.affineNode) # affine node is a child of both rigid and scale node
        # class attributes: API objects
        self.image = None
        self.sampler = None
        self.shapeFunction = None
        # class attributes: sofa components
        self.collision = None # the added collision mesh if any
        self.visual = None # the added visual model if any
        self.rigidDofs = None # rigid dofs
        self.scaleDofs = None # scale dofs
        self.affineDofs = None # affine without shear dofs
        self.mass = None # mass
        self.fixedConstraint = None # to fix the ShearlessAffineBody
        # others class attributes required for several computation
        self.bodyOffset = None # the position of the body
        self.frame = [] # required for many computation, these position are those used to define bones dofs

    def setFromMesh(self, filepath, density=1000, offset=[0,0,0,0,0,0,1], scale3d=[1,1,1], voxelSize=0.01, numberOfPoints=1, generatedDir=None):
        # variables
        self.bodyOffset = Frame.Frame(offset)
        path_affine_rigid = '@' + Tools.node_path_rel(self.affineNode, self.rigidNode)
        path_affine_scale = '@' + Tools.node_path_rel(self.affineNode, self.scaleNode)
        massInfo = SofaPython.mass.RigidMassInfo()
        massInfo.setFromMesh(filepath, density, scale3d)

        self.image = SofaImage.API.Image(self.node, name="image_" + self.name, imageType="ImageUC")
        self.image.node.addChild(self.affineNode)  # for initialization
        self.image.node.addChild(self.rigidNode)  # for initialization
        self.shapeFunction = Flexible.API.ShapeFunction(self.rigidNode)

        if generatedDir is None:
            self.image.addMeshLoader(filepath, value=1, insideValue=1, offset=offset, scale=scale3d) # TODO support multiple meshes closingValue=1,
            self.image.addMeshToImage(voxelSize)
            # rigid dofs
            self.sampler = SofaImage.API.Sampler(self.rigidNode)
            self.sampler.addImageSampler(self.image, numberOfPoints)
            self.rigidDofs = self.sampler.addMechanicalObject('Rigid3'+template_suffix)
        else:
            self.image.addContainer(filename=self.node.name+"_rasterization.raw", directory=generatedDir)
            self.rigidDofs = serialization.importRigidDofs(self.rigidNode, generatedDir+"/"+self.node.name+'_dofs.json')

        # scale dofs
        self.scaleDofs = self.scaleNode.createObject('MechanicalObject', template='Vec3'+template_suffix, name='dofs', position=concat([1,1,1]*numberOfPoints))
        positiveNode = self.scaleNode.createChild('positive')
        positiveNode.createObject('MechanicalObject', template='Vec3'+template_suffix, name='positivescaleDOFs')
        positiveNode.createObject('DifferenceFromTargetMapping', template='Vec3d,Vec3'+template_suffix, applyRestPosition=1, targets=concat(target_scale))
        positiveNode.createObject('UniformCompliance', isCompliance=1, compliance=0)
        positiveNode.createObject('UnilateralConstraint')

        # affine dofs
        self.affineDofs = self.affineNode.createObject('MechanicalObject', template='Affine', name='dofs')
        self.affineNode.createObject('RigidScaleToAffineMultiMapping', template='Rigid,Vec3d,Affine', input1=path_affine_rigid, input2=path_affine_scale, output='@.', autoInit='1', printLog='0')

        # shapefunction and mass
        if generatedDir is None:
            self.shapeFunction.addVoronoi(self.image, position='@dofs.rest_position')
            # mass
            densityImage = self.image.createTransferFunction(self.affineNode, "density", param='0 0 1 '+str(density))
            affineMass = Flexible.API.AffineMass(self.affineNode)
            affineMass.massFromDensityImage(self.affineNode, densityImage=densityImage)
            self.mass = affineMass.mass
        else:
            self.shapeFunction.shapeFunction = serialization.importImageShapeFunction(self.affineNode, generatedDir+self.node.name+"_SF_indices.raw", generatedDir+self.node.name+"_SF_weights.raw", 'dofs')
            self.mass = serialization.importAffineMass(self.affineNode, generatedDir+self.node.name+"_affinemass.json")

        # hack to get the frame position
        self.node.init()
        for p in self.rigidDofs.position:
            p.extend([0,0,0,1])
            self.frame.append(Frame.Frame(p))

    def setManually(self, filepath=None, offset=[[0,0,0,0,0,0,1]], voxelSize=0.01, density=1000, mass=1, inertia=[1,1,1], inertia_forces=False, generatedDir=None):

        if len(offset) == 0:
            Sofa.msg_error("RigidScale.API","ShearlessAffineBody should have at least 1 ShearLessAffine")
            return

        self.framecom = Frame.Frame()
        self.bodyOffset = Frame.Frame([0,0,0,0,0,0,1])
        path_affine_rigid = '@' + Tools.node_path_rel(self.affineNode, self.rigidNode)
        path_affine_scale = '@' + Tools.node_path_rel(self.affineNode, self.scaleNode)
        if len(offset) == 1:
            self.frame = [Frame.Frame(offset[0])]
        str_position = ""
        for p in offset:
            str_position = str_position + concat(p) + " "

        ### scene creation
        # rigid dof
        self.rigidDofs = self.rigidNode.createObject('MechanicalObject', template='Rigid3'+template_suffix, name='dofs', position=str_position, rest_position=str_position)

        # scale dofs
        self.scaleDofs = self.scaleNode.createObject('MechanicalObject', template='Vec3'+template_suffix, name='dofs', position=concat([1,1,1]*len(offset)))
        # The positiveNode is now commented for the moment since it seems not working
        """positiveNode = self.scaleNode.createChild('positive')
        positiveNode.createObject('MechanicalObject', template='Vec3'+template_suffix, name='positivescaleDOFs')
        positiveNode.createObject('DifferenceFromTargetMapping', template='Vec3'+template_suffix+',Vec3'+template_suffix, applyRestPosition=1, targets=concat(target_scale))
        positiveNode.createObject('UniformCompliance', isCompliance=1, compliance=0)
        positiveNode.createObject('UnilateralConstraint')
        positiveNode.createObject('Stabilization', name='Stabilization')"""

        # affine dofs
        self.affineDofs = self.affineNode.createObject('MechanicalObject', template='Affine', name='parent', showObject=0)
        self.affineNode.createObject('RigidScaleToAffineMultiMapping', template='Rigid,Vec3,Affine', input1=path_affine_rigid, input2=path_affine_scale, output='@.', autoInit='1', printLog='0')
        if filepath:
            self.image = SofaImage.API.Image(self.rigidNode, name="image_" + self.name, imageType="ImageUC")
            self.shapeFunction = Flexible.API.ShapeFunction(self.affineNode)
            if generatedDir is None:
                self.image.addMeshLoader(filepath, value=1, insideValue=1)  # TODO support multiple meshes closingValue=1,
                self.image.addMeshToImage(voxelSize)
                self.shapeFunction.addVoronoi(self.image, position='@dofs.rest_position')
                # mass
                self.affineMassNode = self.affineNode.createChild('mass')
                self.affineMassNode.createObject('TransferFunction', name='density', template='ImageUC,ImageD', inputImage='@' + Tools.node_path_rel(self.affineMassNode, self.image.node) + '/image.image', param='0 0 1 '+str(density))
                self.affineMassNode.createObject('MechanicalObject', template='Vec3'+template_suffix)
                self.affineMassNode.createObject('LinearMapping', template='Affine,Vec3'+template_suffix)
                self.affineMassNode.createObject('MassFromDensity',  name='MassFromDensity', template='Affine,ImageD', image='@density.outputImage', transform='@' + Tools.node_path_rel(self.affineMassNode, self.image.node) + '/image.transform', lumping='0')
                self.mass = self.affineNode.createObject('AffineMass', massMatrix='@mass/MassFromDensity.massMatrix')
            else:
                self.image.addContainer(filename=self.node.name + "_rasterization.raw", directory=generatedDir)
                self.shapeFunction.shapeFunction = serialization.importImageShapeFunction(self.affineNode, generatedDir+self.node.name+"_SF_indices.raw", generatedDir+self.node.name+"_SF_weights.raw", 'dofs')
                self.mass = serialization.importAffineMass(self.affineNode, generatedDir+self.node.name+"_affinemass.json")

            # computation of the object mass center
            massInfo = SofaPython.mass.RigidMassInfo()
            massInfo.setFromMesh(filepath, density, [1,1,1])
            # get the object mass center
            self.framecom.rotation = massInfo.inertia_rotation
            self.framecom.translation = massInfo.com
        else:
            if (mass and inertia) and inertia != [0,0,0]:
                Sofa.msg_info("RigidScale", "A RigidMass and a UniformMass are created for respectively the rigid and the scale since there is no mesh which can be used to compute the model mass.")
                self.mass = self.rigidNode.createObject('RigidMass', name='mass', mass=mass, inertia=concat(inertia[:3]), inertia_forces=inertia_forces)
                self.scaleNode.createObject('UniformMass', name='mass', mass=mass)

        self.frame = []
        for o in offset:
            self.frame.append(Frame.Frame(o))

    def setMeshLess(self, offset=[[0,0,0,0,0,0,1]], mass=1, rayleigh=0.1, generatedDir=None):
        if len(offset) == 0:
            Sofa.msg_error("RigidScale.API","ShearlessAffineBody should have at least 1 ShearLessAffine")
            return
        self.framecom = Frame.Frame()
        self.bodyOffset = Frame.Frame([0,0,0,0,0,0,1])
        path_affine_rigid = '@' + Tools.node_path_rel(self.affineNode, self.rigidNode)
        path_affine_scale = '@' + Tools.node_path_rel(self.affineNode, self.scaleNode)
        if len(offset) == 1: self.frame = [Frame.Frame(offset[0])]
        str_position = ""
        for p in offset:
            str_position = str_position + concat(p) + " "

        ### scene creation
        # rigid dof
        self.rigidDofs = self.rigidNode.createObject('MechanicalObject', template='Rigid3'+template_suffix, name='dofs', position=str_position, rest_position=str_position)
        self.rigidNode.createObject('UniformMass', totalMass=mass, rayleighStiffness=rayleigh);

        # scale dofs

        self.scaleDofs = self.scaleNode.createObject('MechanicalObject', template='Vec3'+template_suffix, name='dofs', position= concat([1,1,1]*len(offset)))
        self.scaleNode.createObject('UniformMass', totalMass=mass, rayleighStiffness=rayleigh);
        #positiveNode = self.scaleNode.createChild('positive')
        #positiveNode.createObject('MechanicalObject', template='Vec3'+template_suffix, name='positivescaleDOFs')
        #target_scale = [0.5,0.5,0.5]
        #positiveNode.createObject('DifferenceFromTargetMapping', template='Vec3d,Vec3'+template_suffix, applyRestPosition=1, targets=concat(target_scale))
        #positiveNode.createObject('UniformCompliance', isCompliance=1, compliance=0)
        #positiveNode.createObject('UnilateralConstraint')
        #positiveNode.createObject('Stabilization', name='Stabilization')

        # affine dofs
        self.affineDofs = self.affineNode.createObject('MechanicalObject', template='Affine', name='parent')
        self.affineNode.createObject('RigidScaleToAffineMultiMapping', template='Rigid,Vec3,Affine', input1=path_affine_rigid, input2=path_affine_scale, output='@.', autoInit='1', printLog='0')
        
        self.frame = []
        for o in offset:
            self.frame.append(Frame.Frame(o))

    def addCollisionMesh(self, filepath, scale3d=[1,1,1], offset=[0,0,0,0,0,0,1], name_suffix='', generatedDir=None):
        ## adding a collision mesh to the rigid body with a relative offset
        ## body offset is added to the offset
        # (only a Triangle collision model is created, more models can be added manually)
        self.collision = ShearlessAffineBody.CollisionMesh(self.affineNode, filepath, scale3d, (self.bodyOffset*Frame.Frame(offset)).offset(), name_suffix, generatedDir=generatedDir)
        return self.collision

    def addVisualModel(self, filepath, scale3d=[1,1,1], offset=[0,0,0,0,0,0,1], name_suffix='', generatedDir=None):
        ## adding a visual model to the rigid body with a relative offset
        self.visual = ShearlessAffineBody.VisualModel(self.affineNode, filepath, scale3d, (self.bodyOffset*Frame.Frame(offset)).offset(), name_suffix, generatedDir=generatedDir)
        return self.visual

    def addOffset(self, name, offset=[0,0,0,0,0,0,1], index=-1):
        ## adding a relative offset to the rigid body (e.g. used as a joint location)
        # @warning the translation due to the center of mass offset is automatically removed. If necessary a function without this mecanism could be added
        #return ShearlessAffineBody.Offset(self.rigidNode, self.scaleNode, name, (self.bodyOffset*Frame.Frame(offset)).offset(), index) # this line does not cover the case where the shapeFunction is within the affineNode
        if index > -1:
            return ShearlessAffineBody.Offset(self.rigidNode, self.scaleNode, name, (self.bodyOffset*Frame.frame(offset)).offset(), index)
        else:
            # computation of absolute position of the offset
            offset_abs = self.bodyOffset*Frame.Frame(offset)
            # computation of the index of the closest point to the offset
            ind = 0
            min_dist = sys.float_info.max
            for i, p in enumerate(self.frame):
                dist = numpy.linalg.norm(offset_abs.translation - numpy.array(p.translation), 2)
                if(dist < min_dist):
                    min_dist = dist
                    ind = i
            # add of the offset according to this position
            offset_computed = (self.frame[ind]*offset_abs).offset()
            return ShearlessAffineBody.Offset(self.rigidNode, self.scaleNode, name, offset_computed, ind)

    def addAbsoluteOffset(self, name, offset=[0,0,0,0,0,0,1], index=-1):
        ## adding a offset given in absolute coordinates to the rigid body
        #return ShearlessAffineBody.Offset(self.rigidNode, self.scaleNode, name, offset, index) # this line does not cover the case where the shapeFunction is within the affineNode
        if index > -1:
            return ShearlessAffineBody.Offset(self.rigidNode, self.scaleNode, name, offset, index)
        else :
            # computation of the index of the closest point to the offset
            index_computed = 0
            frameOffset = Frame.Frame(offset)
            min_dist = sys.float_info.max
            for i, p in enumerate(self.frame):
                dist = numpy.linalg.norm(frameOffset.translation - numpy.array(p.translation), 2)
                if(dist < min_dist):
                    min_dist = dist
                    index_computed = i
            # add of the offset according to this position
            offset_computed = frameOffset.offset()
            return ShearlessAffineBody.Offset(self.rigidNode, self.scaleNode, name, offset_computed, index_computed)

    def addMappedPoint(self, name, relativePosition=[0,0,0]):
        ## adding a relative position to the rigid body
        # @warning the translation due to the center of mass offset is automatically removed. If necessary a function without this mecanism could be added
        frame = Frame.Frame(); frame.translation = relativePosition
        return ShearlessAffineBody.MappedPoint(self.affineNode, name, offset)

    def addAbsoluteMappedPoint(self, name, position=[0,0,0]):
        ## adding a position given in absolute coordinates to the rigid body
        frame = Frame.Frame(); frame.translation = position
        return ShearlessAffineBody.MappedPoint(self.affineNode, name, offset)

    def addMotor(self, forces=[0,0,0,0,0,0]):
        ## adding a constant force/torque to the rigid body (that could be driven by a controller to simulate a motor)
        return self.rigidNode.createObject('ConstantForceField', template='Rigid3'+template_suffix, name='motor', points='0', forces=concat(forces))

    def addBehavior(self, youngModulus=1E2, numberOfGaussPoint=100, generatedDir=None):
        ## adding behavior to the component
        self.behavior = Flexible.API.Behavior(self.affineNode, "behavior", type="331")
        self.behavior.addGaussPointSampler(self.shapeFunction, numberOfGaussPoint)
        self.behavior.addMechanicalObject(dofAffineNode=self.affineNode)
        #self.behavior.addHooke(youngModulus=youngModulus)
        self.behavior.addProjective(youngModulus=youngModulus) # , poissonRatio=0
        return self.behavior

    def setFixed(self, isFixed=True):
        ''' Add/remove a fixed constraint for this rigid '''
        if isFixed and self.fixedConstraint is None:
            self.fixedConstraint = self.rigidNode.createObject('FixedConstraint', name='fixedConstraint')
        elif not isFixed and not self.fixedConstraint is None:
            self.rigidNode.removeObject(self.fixedConstraint)
            self.fixedConstraint=None

    class CollisionMesh:

        def __init__(self, node, filepath, scale3d, offset, name_suffix='', generatedDir=None):
            r = Quaternion.to_euler(offset[3:]) * 180.0 / math.pi
            self.node = node.createChild('collision'+name_suffix)  # node
            self.loader = SofaPython.Tools.meshLoader(self.node, filename=filepath, name='loader', scale3d=concat(scale3d), translation=concat(offset[:3]) , rotation=concat(r), triangulate=True)
            self.topology = self.node.createObject('MeshTopology', name='topology', src='@loader')
            self.dofs = self.node.createObject('MechanicalObject', name='dofs', template='Vec3'+template_suffix)
            self.triangles = self.node.createObject('TriangleModel', name='model')
            if generatedDir is None:
                self.mapping = self.node.createObject('LinearMapping', template='Affine,Vec3'+template_suffix, name='mapping')
            else:
                serialization.importLinearMapping(self.node, generatedDir+"_collisionmapping.json")
            self.normals = None

        def addNormals(self, invert=False):
            ## add a component to compute mesh normals at each timestep
            self.normals = self.node.createObject('NormalsFromPoints', template='Vec3'+template_suffix, name='normalsFromPoints', position='@'+self.dofs.name+'.position', triangles='@'+self.topology.name+'.triangles', quads='@'+self.topology.name+'.quads', invertNormals=invert)

        def addVisualModel(self):
            ## add a visual model identical to the collision model
            return ShearlessAffineBody.CollisionMesh.VisualModel(self.node)

        class VisualModel:

            def __init__(self, node):
                global idxVisualModel;
                self.node = node.createChild('visual')  # node
                self.model = self.node.createObject('VisualModel', name='model'+str(idxVisualModel)) # @to do: Add the filename in order to keep the texture coordinates otherwise we lost them ...
                self.mapping = self.node.createObject('IdentityMapping', name='mapping')
                idxVisualModel+=1

    class VisualModel:

        def __init__(self, node, filepath, scale3d, offset, name_suffix='', generatedDir=None):
            r = Quaternion.to_euler(offset[3:]) * 180.0 / math.pi
            global idxVisualModel;
            self.node = node.createChild('visual'+name_suffix)  # node
            self.model = self.node.createObject('VisualModel', name='visual'+str(idxVisualModel), fileMesh=filepath, scale3d=concat(scale3d), translation=concat(offset[:3]), rotation=concat(r))
            if generatedDir is None:
                self.mapping = self.node.createObject('LinearMapping', template='Affine,ExtVec3f', name='mapping')
            else:
                serialization.importLinearMapping(self.node, generatedDir+"_visualmapping.json")
            idxVisualModel+=1

    class Offset:

        def __init__(self, rigidNode, scaleNode, name, offset, arg=-1):
            # node creation
            self.node = rigidNode.createChild(name)
            scaleNode.addChild(self.node)
            # variables
            self.frame = Frame.Frame(offset)
            path_offset_rigid = '@' + Tools.node_path_rel(self.node, rigidNode)
            path_offset_scale = '@' + Tools.node_path_rel(self.node, scaleNode)
            # scene creation
            self.dofs = self.frame.insert(self.node, template='Rigid3'+template_suffix, name='dofs')

            if arg==-1:
                self.mapping = self.node.createObject('RigidScaleToRigidMultiMapping', template='Rigid3'+template_suffix+',Vec3'+template_suffix+',Rigid3'+template_suffix
                                                                                     , input1=path_offset_rigid, input2=path_offset_scale, output='@.'
                                                                                     , useGeometricStiffness=geometric_stiffness, printLog='0')
            else:
                self.mapping = self.node.createObject('RigidScaleToRigidMultiMapping', template='Rigid3'+template_suffix+',Vec3'+template_suffix+',Rigid3'+template_suffix
                                                                                     , input1=path_offset_rigid, input2=path_offset_scale, output='@.'
                                                                                     , index='0 '+ str(arg) + ' ' + str(arg), useGeometricStiffness=geometric_stiffness, printLog='0')

        def addOffset(self, name, offset=[0,0,0,0,0,0,1]):
            ## adding a relative offset to the offset
            return ShearlessAffineBody.Offset(self.node, name, offset)

        def addAbsoluteOffset(self, name, offset=[0,0,0,0,0,0,1]):
            ## adding a offset given in absolute coordinates to the offset
            return ShearlessAffineBody.Offset(self.node, name, (Frame.Frame(offset) * self.frame.inv()).offset())

        def addMotor(self, forces=[0,0,0,0,0,0]):
            ## adding a constant force/torque at the offset location (that could be driven by a controller to simulate a motor)
            return self.node.createObject('ConstantForceField', template='Rigid3'+template_suffix, name='motor', points='0', forces=concat(forces))

        def addMappedPoint(self, name, relativePosition=[0,0,0]):
            ## adding a relative position to the rigid body
            return ShearlessAffineBody.Offset.MappedPoint(self.node, name, relativePosition)

        def addAbsoluteMappedPoint(self, name, position=[0,0,0]):
            ## adding a position given in absolute coordinates to the rigid body
            frame = Frame.Frame(); frame.translation = position
            return ShearlessAffineBody.Offset.MappedPoint(self.node, name, (frame * self.frame.inv()).translation)

        class MappedPoint:

            def __init__(self, node, name, position):
                self.node = node.createChild(name)
                self.dofs = self.node.createObject('MechanicalObject', name='dofs', template='Vec3'+template_suffix, position=concat(position))
                self.mapping = self.node.createObject('RigidMapping', name='mapping', geometricStiffness = geometric_stiffness)

    class MappedPoint:

        def __init__(self, node, name, position):
            self.node = node.createChild(name)
            self.dofs = self.node.createObject('MechanicalObject', name='dofs', template='Vec3'+template_suffix, position=concat(position))
            self.mapping = self.node.createObject('LinearMapping', name='mapping', geometricStiffness = geometric_stiffness)

    # Export of class
    def exportRasterization(self, path="./generated"):
        self.image.addExporter(filename=self.node.name+"_rasterization.raw", directory=path)

    def exportShapeFunction(self, path="./generated"):
        serialization.exportImageShapeFunction(self.affineNode, self.shapeFunction.shapeFunction, path+"/"+self.node.name+"_SF_indices.raw", path+"/"+self.node.name+"_SF_weights.raw")

    def exportMappings(self, path="./generated"):
        if self.collision is not None:
            serialization.exportLinearMapping(self.collision.mapping, path+"/"+self.node.name+'_collisionmapping.json')
        if self.visual is not None:
            serialization.exportLinearMapping(self.visual.mapping, path+"/"+self.node.name+'_visualmapping.json')
        if self.behavior is not None:
            serialization.exportLinearMapping(self.behavior.mapping, path+"/"+self.node.name+'_behaviormapping.json')

    def exportGaussPoints(self, path="./generated"):
        if self.behavior is not None:
            serialization.exportGaussPoints(self.behavior.sampler, path+"/"+self.node.name+'_gauss.json')

    def exportAffineMass(self, path="./generated"):
        serialization.exportAffineMass(self.mass, path+"/"+self.node.name+'_affinemass.json')

    def exportRigidDofs(self, path="./generated"):
        serialization.exportRigidDofs(self.rigidDofs, path+"/"+self.node.name+'_dofs.json')
