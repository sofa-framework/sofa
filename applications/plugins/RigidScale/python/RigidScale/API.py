#from operator import ge

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
import Flexible.Serialization

# to specify the floating point encoding (double by default)
template_suffix=''

# global variable to give a different name to each visual model
idxVisualModel = 0

# to use geometric_stiffness of rigid mappings
# @warning WIP, the API will change
geometric_stiffness = 0

# target scale used when the scale is close to 0
target_scale = [1E-6, 1E-6, 1E-6]

class ShearlessAffineBody:
    
    # Generic Body composed by one or more affine without shear
    def __init__(self, node, name, numberOfPoints=1):
        # node creation
        self.node = node.createChild(name)
        self.rigidNode = self.node.createChild(name + '_rigid')  # rigid node
        self.scaleNode = self.node.createChild(name + '_scale')  # scale node
        self.affineNode = self.rigidNode.createChild(name + '_affine')  # affine node is a child of both rigid and scale node
        self.scaleNode.addChild(self.affineNode) # affine node is a child of both rigid and scale node
        # class attributes: sofa components
        self.collision = None # the added collision mesh if any
        self.visual = None # the added visual model if any
        self.rigidDofs = None # rigid dofs
        self.scaleDofs = None # scale dofs
        self.affineDofs = None # affine without shear dofs
        self.mass = None # mass
        self.fixedConstraint = None # to fix the ShearlessAffineBody
        # others class attributes required for several computation
        self.frame = [] # required for many computation, these position are those used to define bones dofs
        self.framecom = Frame.Frame() # frame at the center of mass
        self.numberOfPoints = numberOfPoints # number of controlling a bone

    def setFromMesh(self, filepath, density=1000, offset=[0,0,0,0,0,0,1], scale3d=[1,1,1], inertia_forces=False, voxelSize=0.01, generatedDir=None):
        # variables
        r = Quaternion.to_euler(offset[3:]) * 180.0 / math.pi
        path_affine_rigid = '@'+ Tools.node_path_rel(self.affineNode, self.rigidNode)
        path_affine_scale = '@'+ Tools.node_path_rel(self.affineNode, self.scaleNode)
        massInfo = SofaPython.mass.RigidMassInfo()
        massInfo.setFromMesh(filepath, density, scale3d)
        # get the object mass center
        self.framecom = Frame.Frame()
        self.framecom.translation = massInfo.com
        self.framecom.rotation = massInfo.inertia_rotation
        self.framecom = Frame.Frame(offset) * self.framecom
        if self.numberOfPoints == 1:
            self.frame = [self.framecom]
            self.setFromRigidInfo(massInfo, offset, inertia_forces)
            # shape function
            self.affineNode.createObject('MeshObjLoader',name='source', filename=filepath, triangulate=1, scale3d=concat(scale3d), translation=concat(offset[:3]) , rotation=concat(r))

            if generatedDir is None:
                self.meshToImageEngine = self.affineNode.createObject('MeshToImageEngine', template='ImageUC', name='rasterizer', src='@source', value=1, insideValue=1, voxelSize=voxelSize, padSize=0, rotateImage='false')
                self.affineNode.createObject('ImageContainer', template='ImageUC', name='image', src='@rasterizer')
                self.shapeFunction=self.affineNode.createObject('VoronoiShapeFunction', template='ShapeFunctiond,ImageUC', name='SF', position='@dofs.rest_position', image='@image.image', transform='@image.transform', nbRef=8, clearData=1, bias=0)
            else:
                self.affineNode.createObject('ImageContainer', template='ImageUC', name='image', filename=generatedDir+self.node.name+"_rasterization.raw", drawBB='false')
                serialization.importImageShapeFunction( self.affineNode,generatedDir+self.node.name+"_SF_indices.raw",generatedDir+self.node.name+"_SF_weights.raw", 'dofs' )
        else:
            # variables
            rigid_mass = ' '
            rigid_inertia = ' '
            scale_rest_position = ''
            for i in range(self.numberOfPoints):
                rigid_mass = rigid_mass + ' ' + str(massInfo.mass)
                rigid_inertia = rigid_inertia + ' ' + concat(massInfo.diagonal_inertia)
                scale_rest_position = scale_rest_position + ' ' + concat([1,1,1])
            # rigid dofs
            meshLoaderComponent = self.rigidNode.createObject('MeshObjLoader',name='source', filename=filepath, triangulate=1, scale3d=concat(scale3d), translation=concat(offset[:3]) , rotation=concat(r))

            if generatedDir is None:
                self.meshToImageEngine = self.rigidNode.createObject('MeshToImageEngine', template='ImageUC', name='rasterizer', src='@source', value=1, insideValue=1, voxelSize=voxelSize, padSize=0, rotateImage='false')
                imageContainerComponent = self.rigidNode.createObject('ImageContainer', template='ImageUC', name='image', src='@rasterizer')
            else:
                imageContainerComponent = self.rigidNode.createObject('ImageContainer', template='ImageUC', name='image', filename=generatedDir+self.node.name+"_rasterization.raw", drawBB='false')

            if generatedDir is None:
                imageSamplerComponent = self.rigidNode.createObject('ImageSampler', template='ImageUC', name='sampler', src='@image', method=1, param=str(self.numberOfPoints), clearData=1)
                self.rigidDofs = self.rigidNode.createObject('MechanicalObject', template='Rigid3'+template_suffix, name='dofs', position='@sampler.position')
            else:
                self.rigidDofs = serialization.importRigidDofs(self.rigidNode,generatedDir+self.node.name+"_dofs.json")

            # scale dofs
            self.scaleDofs = self.scaleNode.createObject('MechanicalObject', template='Vec3'+template_suffix, name='dofs', position=scale_rest_position)
            positiveNode = self.scaleNode.createChild('positive')
            positiveNode.createObject('MechanicalObject', template='Vec3'+template_suffix, name='positivescaleDOFs')
            positiveNode.createObject('DifferenceFromTargetMapping', template='Vec3d,Vec3'+template_suffix, applyRestPosition=1, targets=concat(target_scale))
            positiveNode.createObject('UniformCompliance', isCompliance=1, compliance=0)
            positiveNode.createObject('UnilateralConstraint')
            positiveNode.createObject('Stabilization', name='Stabilization')
            # affine dofs
            self.affineDofs = self.affineNode.createObject('MechanicalObject', template='Affine', name='dofs', showObject=1, showObjectScale='0.1')
            self.affineNode.createObject('RigidScaleToAffineMultiMapping', template='Rigid,Vec3d,Affine', input1=path_affine_rigid, input2=path_affine_scale, output='@.', autoInit='1', printLog='0')


            if generatedDir is None:
                self.affineNode.createObject('ImageContainer', template='ImageUC', name='image', src=path_affine_rigid+'/rasterizer')
                self.shapeFunction=self.affineNode.createObject('VoronoiShapeFunction', template='ShapeFunctiond,ImageUC', name='SF', position='@dofs.rest_position', image='@image.image', transform='@image.transform', nbRef=8, clearData=1, bias=0)
                self.meshToImageEngine.init()
                imageSamplerComponent.init()
            else:
                self.shapeFunction=serialization.importImageShapeFunction( self.affineNode, generatedDir+self.node.name+"_SF_indices.raw",generatedDir+self.node.name+"_SF_weights.raw", 'dofs' )


            # init of component for being able to acces to bones position
            imageContainerComponent.init()
            # acces to position
            self.frame = []
            # @warning the position in imageSamplerComponent.position are computed without the offset applying in the mesh loader
            for t in self.rigidDofs.position:
                p = t
                self.frame.append(Frame.Frame(offset) * Frame.Frame(p))

        # mass
        if generatedDir is None:
            self.affineMassNode = self.affineNode.createChild('mass')
            self.affineMassNode.createObject('TransferFunction', name='density', template='ImageUC,ImageD', inputImage='@../image.image', param='0 0 1 '+str(density))
            self.affineMassNode.createObject('MechanicalObject', template='Vec3'+template_suffix)
            self.affineMassNode.createObject('LinearMapping', template='Affine,Vec3'+template_suffix)
            self.affineMassNode.createObject('MassFromDensity',  name='MassFromDensity', template='Affine,ImageD', image='@density.outputImage', transform='@../image.transform', lumping='0')
            self.affineMass = self.affineNode.createObject('AffineMass', massMatrix='@mass/MassFromDensity.massMatrix')
        else:
            serialization.importAffineMass(self.affineNode,generatedDir+self.node.name+"_affinemass.json")
        return

    def setManually(self, filepath=None, offset=[[0,0,0,0,0,0,1]], mass=[1], inertia=[[1,1,1]], inertia_forces=False, voxelSize=0.01, density=2000, generatedDir=None):
        if len(offset) == 0:
            print 'StructuralAPIShearlessAffine: The case the number of points per bones equal ' + str(self.numberOfPoints) + 'is not yet handled.'
            return
        self.framecom = Frame.Frame()
        path_affine_rigid = '@'+ Tools.node_path_rel(self.affineNode, self.rigidNode)
        path_affine_scale = '@'+ Tools.node_path_rel(self.affineNode, self.scaleNode)
        if self.numberOfPoints == 1: self.frame = [Frame.Frame(offset[0])]
        rigid_inertia = ' '
        scale_rest_position = ''
        for m in inertia:
            rigid_inertia = rigid_inertia + ' ' + concat(m)
        for i in range(self.numberOfPoints):
            scale_rest_position = scale_rest_position + ' ' + concat([1,1,1])
        str_position = ""
        for p in offset:
            str_position = str_position + concat(p) + " "
        # scene creation
        # rigid dof
        self.rigidDofs = self.rigidNode.createObject('MechanicalObject', template='Rigid3'+template_suffix, name='dofs', position=str_position, rest_position=str_position)

        # scale dofs
        self.scaleDofs = self.scaleNode.createObject('MechanicalObject', template='Vec3'+template_suffix, name='dofs', position=scale_rest_position)
        positiveNode = self.scaleNode.createChild('positive')
        positiveNode.createObject('MechanicalObject', template='Vec3'+template_suffix, name='positivescaleDOFs')
        positiveNode.createObject('DifferenceFromTargetMapping', template='Vec3'+template_suffix+',Vec3'+template_suffix, applyRestPosition=1, targets=concat(target_scale))
        positiveNode.createObject('UniformCompliance', isCompliance=1, compliance=0)
        positiveNode.createObject('UnilateralConstraint')
        positiveNode.createObject('Stabilization', name='Stabilization')

        # affine dofs
        self.affineDofs = self.affineNode.createObject('MechanicalObject', template='Affine', name='parent', showObject=0)
        self.affineNode.createObject('RigidScaleToAffineMultiMapping', template='Rigid,Vec3,Affine', input1=path_affine_rigid, input2=path_affine_scale, output='@.', autoInit='1', printLog='0')
        if filepath:
            self.affineNode.createObject('MeshObjLoader',name='source', filename=filepath, triangulate=1)
            if generatedDir is None:
                self.meshToImageEngine = self.affineNode.createObject('MeshToImageEngine', template='ImageUC', name='rasterizer', src='@source', value=1, insideValue=1, voxelSize=voxelSize, padSize=0, rotateImage='false')
                self.affineNode.createObject('ImageContainer', template='ImageUC', name='image', src='@rasterizer')
                self.shapeFunction=self.affineNode.createObject('VoronoiShapeFunction', template='ShapeFunctiond,ImageUC', name='SF', position='@dofs.rest_position', image='@image.image', transform='@image.transform', nbRef=8, clearData=1, bias=0)
            else:
                self.affineNode.createObject('ImageContainer', template='ImageUC', name='image', filename=generatedDir+self.node.name+"_rasterization.raw", drawBB='false')
                serialization.importImageShapeFunction( self.affineNode,generatedDir+self.node.name+"_SF_indices.raw", generatedDir+self.node.name+"_SF_weights.raw", 'dofs' )

            # mass
            if generatedDir is None:
                self.affineMassNode = self.affineNode.createChild('mass')
                self.affineMassNode.createObject('TransferFunction', name='density', template='ImageUC,ImageD', inputImage='@../image.image', param='0 0 1 '+str(density))
                self.affineMassNode.createObject('MechanicalObject', template='Vec3'+template_suffix)
                self.affineMassNode.createObject('LinearMapping', template='Affine,Vec3'+template_suffix)
                self.affineMassNode.createObject('MassFromDensity',  name='MassFromDensity', template='Affine,ImageD', image='@density.outputImage', transform='@../image.transform', lumping='0')
                self.affineMass = self.affineNode.createObject('AffineMass', massMatrix='@mass/MassFromDensity.massMatrix')
            else:
                serialization.importAffineMass(self.affineNode,generatedDir+self.node.name+"_affinemass.json")

            # computation of the object mass center
            massInfo = SofaPython.mass.RigidMassInfo()
            massInfo.setFromMesh(filepath, density, [1,1,1])
            # get the object mass center
            self.framecom.rotation = massInfo.inertia_rotation
            self.framecom.translation = massInfo.com
        else:
            print "You need a mesh to create an articulated system"
        self.frame = []
        for o in offset:
            self.frame.append(Frame.Frame(o))

    def setFromRigidInfo(self, info, offset=[0,0,0,0,0,0,1], inertia_forces=False):
        # variables
        path_affine_rigid = '@'+ Tools.node_path_rel(self.affineNode, self.rigidNode)
        path_affine_scale = '@'+ Tools.node_path_rel(self.affineNode, self.scaleNode)
        # rigid dofs
        self.rigidDofs = self.frame[0].insert(self.rigidNode, name = 'dofs', template='Rigid3'+template_suffix)
        # scale dofs
        self.scaleDofs = self.scaleNode.createObject('MechanicalObject', template='Vec3'+template_suffix, name='dofs', position='1 1 1')
        positiveNode = self.scaleNode.createChild('positive')
        positiveNode.createObject('MechanicalObject', template='Vec3'+template_suffix, name='positivescaleDOFs')
        positiveNode.createObject('DifferenceFromTargetMapping', template='Vec3'+template_suffix+',Vec3'+template_suffix, applyRestPosition=1, targets=concat(target_scale))
        positiveNode.createObject('UniformCompliance', isCompliance=1, compliance=0)
        positiveNode.createObject('UnilateralConstraint')
        positiveNode.createObject('Stabilization', name='Stabilization')
        # affine dofs
        self.affineDofs = self.affineNode.createObject('MechanicalObject', template='Affine', name='parent', showObject=0)
        self.affineNode.createObject('RigidScaleToAffineMultiMapping', template='Rigid,Vec3d,Affine', input1=path_affine_rigid, input2=path_affine_scale, output='@.', autoInit='1', printLog='0')
        return

    def addCollisionMesh(self, filepath, scale3d=[1,1,1], offset=[0,0,0,0,0,0,1], name_suffix='', generatedDir=None):
        ## adding a collision mesh to the rigid body with a relative offset
        # (only a Triangle collision model is created, more models can be added manually)
        # @warning the translation due to the center of mass offset is automatically removed. If necessary a function without this mecanism could be added
        self.collision = ShearlessAffineBody.CollisionMesh(self.affineNode, filepath, scale3d, offset, name_suffix, generatedDir=generatedDir)
        return self.collision

    def addVisualModel(self, filepath, scale3d=[1,1,1], offset=[0,0,0,0,0,0,1], name_suffix='', generatedDir=None):
        ## adding a visual model to the rigid body with a relative offset
        # @warning the translation due to the center of mass offset is automatically removed. If necessary a function without this mecanism could be added
        self.visual = ShearlessAffineBody.VisualModel(self.affineNode, filepath, scale3d, offset, name_suffix, generatedDir=generatedDir)
        return self.visual

    def addOffset(self, name, offset=[0,0,0,0,0,0,1], index=-1): #TODO None value instead of -1
        ## adding a relative offset to the rigid body (e.g. used as a joint location)
        # @warning the translation due to the center of mass offset is automatically removed. If necessary a function without this mecanism could be added
        if index > -1:
            return ShearlessAffineBody.Offset(self.rigidNode, name, offset, index)
        if self.numberOfPoints == 1:
            return ShearlessAffineBody.Offset(self.rigidNode, name, offset, 0)
        else:
            # computation of absolute position of the offset
            offset_abs = self.framecom*Frame.Frame(offset)
            # computation of the index of the closest point to the offset
            ind = 0
            min_dist = numpy.linalg.norm(numpy.array(offset_abs.translation) - numpy.array(self.frame[0].translation), 2)
            for i, p in enumerate(self.frame):
                dist = numpy.linalg.norm(numpy.array(offset_abs.translation) - numpy.array(p.translation), 2)
                if(dist < min_dist):
                    min_dist = dist
                    ind = i
            # add of the offset according to this position
            offset = (self.frame[ind].inv()*offset_abs).offset()
            return ShearlessAffineBody.Offset(self.rigidNode, name, offset, ind)

    def addAbsoluteOffset(self, name, offset=[0,0,0,0,0,0,1], index=-1):
        ## adding a offset given in absolute coordinates to the rigid body
        if index > -1:
            return ShearlessAffineBody.Offset(self.rigidNode, name, (self.frame[index].inv()*Frame.Frame(offset)).offset(), index)
        if self.numberOfPoints == 1:
            offset = (self.frame[0].inv()*Frame.Frame(offset)).offset()
            return ShearlessAffineBody.Offset(self.rigidNode, name, offset, 0)
        else:
            # computation of the index of the closest point to the offset
            ind = 0
            offset = Frame.Frame(offset)
            min_dist = numpy.linalg.norm(numpy.array(offset.translation) - numpy.array(self.frame[0].translation), 2)
            for i, p in enumerate(self.frame):
                dist = numpy.linalg.norm(numpy.array(offset.translation) - numpy.array(p.translation), 2)
                if(dist < min_dist):
                    min_dist = dist
                    ind = i
            # add of the offset according to this position
            offset = (self.frame[ind].inv()*offset).offset()
            return ShearlessAffineBody.Offset(self.rigidNode, name, offset, ind)

    def addRigidScaleOffset(self, name, offset=[0,0,0,0,0,0,1], index=-1):
        ## adding a relative offset to the rigid body (e.g. used as a joint location)
        # @warning the translation due to the center of mass offset is automatically removed. If necessary a function without this mecanism could be added
        if index > -1:
            return ShearlessAffineBody.RigidScaleOffset(self.rigidNode, self.scaleNode, name, offset, index)
        if self.numberOfPoints == 1:
            return ShearlessAffineBody.RigidScaleOffset(self.rigidNode, self.scaleNode, name, offset, 0)
        else:
            # computation of absolute position of the offset
            offset_abs = self.framecom*Frame.Frame(offset)
            # computation of the index of the closest point to the offset
            ind = 0
            min_dist = numpy.linalg.norm(numpy.array(offset_abs.translation) - numpy.array(self.frame[0].translation), 2)
            for i, p in enumerate(self.frame):
                dist = numpy.linalg.norm(numpy.array(offset_abs.translation) - numpy.array(p.translation), 2)
                if(dist < min_dist):
                    min_dist = dist
                    ind = i
            # add of the offset according to this position
            offset = (self.frame[ind].inv()*offset_abs).offset()
            return ShearlessAffineBody.RigidScaleOffset(self.rigidNode, self.scaleNode, name, offset, ind)

    def addAbsoluteRigidScaleOffset(self, name, offset=[0,0,0,0,0,0,1], index=-1):
        ## adding a offset given in absolute coordinates to the rigid body
        if index > -1:
            return ShearlessAffineBody.RigidScaleOffset(self.rigidNode, self.scaleNode, name, (self.frame[index].inv()*Frame.Frame(offset)).offset(), index)
        if self.numberOfPoints == 1:
            offset = (self.frame[0].inv()*Frame.Frame(offset)).offset()
            return ShearlessAffineBody.RigidScaleOffset(self.rigidNode, self.scaleNode, name, offset, 0)
        elif self.numberOfPoints > 1:
            # computation of the index of the closest point to the offset
            index_computed = 0
            offset = Frame.Frame(offset)
            min_dist = numpy.linalg.norm(numpy.array(offset.translation) - numpy.array(self.frame[0].translation), 2)
            for i, p in enumerate(self.frame):
                dist = numpy.linalg.norm(numpy.array(offset.translation) - numpy.array(p.translation), 2)
                if(dist < min_dist):
                    min_dist = dist
                    index_computed = i
            # add of the offset according to this position
            offset_computed = (self.frame[index_computed].inv()*offset).offset()
            return ShearlessAffineBody.RigidScaleOffset(self.rigidNode, self.scaleNode, name, offset_computed, index_computed)

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

    def addElasticBehavior(self, name, stiffness=1E2, poissonRatio=0, numberOfGaussPoint=100, generatedDir=None):
        ## adding elastic behavior to the component
        self.behavior = ShearlessAffineBody.ElasticBehavior(self.affineNode, name, stiffness, poissonRatio, numberOfGaussPoint, generatedDir=generatedDir)
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
            r = Quaternion.to_euler(offset[3:])  * 180.0 / math.pi
            self.node = node.createChild('collision'+name_suffix)  # node
            self.loader = SofaPython.Tools.meshLoader(self.node, filename=filepath, name='loader', scale3d=concat(scale3d), translation=concat(offset[:3]) , rotation=concat(r), triangulate=True)
            self.topology = self.node.createObject('MeshTopology', name='topology', src='@loader')
            self.dofs = self.node.createObject('MechanicalObject', name='dofs', template='Vec3'+template_suffix)
            self.triangles = self.node.createObject('TriangleModel', name='model')
            if generatedDir is None:
                self.mapping = self.node.createObject('LinearMapping', template='Affine,Vec3'+template_suffix, name='mapping')
            else:
                serialization.importLinearMapping(self.node,generatedDir+"_collisionmapping.json")
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
                self.model = self.node.createObject('VisualModel', name='model'+str(idxVisualModel))
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
                serialization.importLinearMapping(self.node,generatedDir+"_visualmapping.json")
            idxVisualModel+=1

    class Offset:

        def __init__(self, rigidNode, name, offset, arg=0):
            self.node = rigidNode.createChild(name)
            self.frame = Frame.Frame(offset)
            self.dofs = self.frame.insert(self.node, name='dofs', template='Rigid3'+template_suffix)
            self.mapping = self.node.createObject('AssembledRigidRigidMapping', name='mapping', source = str(arg)+' '+str(self.frame), geometricStiffness=geometric_stiffness)

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

    class RigidScaleOffset:

        def __init__(self, rigidNode, scaleNode, name, offset, arg=0):
            # node creation
            self.node = rigidNode.createChild(name)
            scaleNode.addChild(self.node)
            # variables
            self.frame = Frame.Frame(offset)
            path_offset_rigid = '@'+ Tools.node_path_rel(self.node, rigidNode)
            path_offset_scale = '@'+ Tools.node_path_rel(self.node, scaleNode)
            # scene creation
            self.dofs = self.frame.insert(self.node, name='dofs', template='Rigid3'+template_suffix)
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
            return ShearlessAffineBody.RigidScaleOffset.MappedPoint(self.node, name, relativePosition)

        def addAbsoluteMappedPoint(self, name, position=[0,0,0]):
            ## adding a position given in absolute coordinates to the rigid body
            frame = Frame.Frame(); frame.translation = position
            return ShearlessAffineBody.RigidScaleOffset.MappedPoint(self.node, name, (frame * self.frame.inv()).translation)

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

    class ElasticBehavior:

        def __init__(self, node, name, stiffness=1E2, poissonCoef=0, numberOfGaussPoint=100, generatedDir=None):
            self.node = node.createChild(name)
            if generatedDir is None:
                self.sampler = self.node.createObject('ImageGaussPointSampler', name='sampler', indices='@../SF.indices', weights='@../SF.weights', transform='@../SF.transform', method=2, order=1, targetNumber=numberOfGaussPoint)
            else:
                serialization.importGaussPoints(self.node,generatedDir+"_gauss.json")
            self.dofs = self.node.createObject('MechanicalObject', template='F331')
            if generatedDir is None:
                self.mapping = self.node.createObject('LinearMapping', template='Affine,F331')
            else:
                serialization.importLinearMapping(self.node,generatedDir+"_behaviormapping.json")
            self.forcefield = self.node.createObject('ProjectiveForceField', template='F331', youngModulus=stiffness, poissonRatio=poissonCoef, viscosity=0, isCompliance=0)
            #strainNode = self.node.createChild('strain')
            #strainNode.createObject('MechanicalObject', template="E331", name="E")
            #strainNode.createObject('CorotationalStrainMapping', template="F331,E331", method="svd", geometricStiffness=0)
            #self.forcefield = strainNode.createObject('HookeForceField', template="E331", name="forcefield", youngModulus=stiffness, poissonRatio=poissonCoef, viscosity="0.0")



    def exportRasterization(self,path="./generated"):
        self.meshToImageEngine.getContext().createObject('ImageExporter', name="writer", src="@rasterizer", filename=path+"/"+self.node.name+"_rasterization.raw", exportAtBegin="true")

    def exportShapeFunction(self,path="./generated"):
        serialization.exportImageShapeFunction(self.affineNode,self.shapeFunction,path+"/"+self.node.name+"_SF_indices.raw",path+"/"+self.node.name+"_SF_weights.raw")

    def exportMappings(self,path="./generated"):
        if self.collision is not None:
            serialization.exportLinearMapping(self.collision.mapping,path+"/"+self.node.name+'_collisionmapping.json')
        if self.visual is not None:
            serialization.exportLinearMapping(self.visual.mapping,path+"/"+self.node.name+'_visualmapping.json')
        if self.behavior is not None:
            serialization.exportLinearMapping(self.behavior.mapping,path+"/"+self.node.name+'_behaviormapping.json')

    def exportGaussPoints(self,path="./generated"):
        if self.behavior is not None:
            serialization.exportGaussPoints(self.behavior.sampler,path+"/"+self.node.name+'_gauss.json')

    def exportAffineMass(self,path="./generated"):
        serialization.exportAffineMass(self.affineMass,path+"/"+self.node.name+'_affinemass.json')

    def exportRigidDofs(self,path="./generated"):
        serialization.exportRigidDofs(self.rigidDofs,path+"/"+self.node.name+'_dofs.json')
