import os, sys, math, random

import numpy

import tools

import SofaPython.Tools
import SofaPython.units
from SofaPython import Quaternion
from SofaPython.Tools import listToStr as listToStr, listListToStr as listListToStr
import SofaPython.mass
import SofaPython.Quaternion
import Flexible.sml, SofaPython.sml

from collections import OrderedDict as odict

from Compliant import Frame as Frame, Tools as Tools, StructuralAPI

import RigidScale.API

# Recognized flag from the sml
BoneType = ("flat", "irregular", "long", "short", "special")
FramePerBoneType = {"short": 1, "long": 2, "flat": 3, "irregular": 3, "special": 20}
IntegrationPointPerBoneType = {"short": 1, "long": 8, "flat": 12, "irregular": 12, "special": 20}
TransferMethod = ("minimal", "interpolation", "classic", "icp")
ConstraintType = ("straight")
ROIFlag = ("frameROI", "registrationROI")

straight_constraint_default_frame = [0, 0, 0, 0, 0, 0, 1]
straight_constraint_default_mask = [0, 1, 1, 1, 1, 1]
straight_constraint_default_limit = [-10,10, 0,0, 0,0, 0,0, 0,0, 0,0]

## ==============================================================================
## @Description: Bone for registration
## Each bone ca be composed by one or many frames, each frame is an affine frame
## without any possibility to shear.
"""
 - @param: name: bone name
 - @param: type: type bone (long, short, flat, irregular), this information will be used to
         set the number of frame per bone, and the bone will be transfered.
 - @param: filepath: path of the bone geometry
 - @param: frame: list of frame (position + quaternion) composing the bone
 - @param: elasticity: since bone is used for registration, bone needs to be a deformable
 object
 - @param: voxel_size: the size of voxel used to rasterize object, to create the shape function
"""
## ==============================================================================
class Bone():

    def __init__(self, name='', type=None, filepath=None, frame=None, transform=[0,0,0,0,0,0,1,1,1]):

        # specific attributes
        self.name = name
        self.type = type
        self.filepath = filepath
        self.frame = frame
        self.inertia = []
        self.transform = transform
        self.body = None
        self.elasticity = None
        self.voxelSize = None

        # sofa components
        self.behavior = None
        self.collision = None
        self.visual = None

    def __str__(self):
        str = ""
        str =  str + "Bone: " + self.name + "\n"
        str =  str + "--- type -> " + self.type + "\n"
        str =  str + "--- filepath -> " + self.filepath + "\n"
        str =  str + "--- frame -> " + listToStr(self.frame) + "\n"
        str =  str + "--- inertia -> " + listToStr(self.inertia) + "\n"
        str =  str + "--- transform -> " + listToStr(self.transform) + "\n"
        str =  str + "--- voxelSize -> " + repr(self.voxelSize)
        return str

    def initUsingSMLModel(self, model):
        # Let's set all the parameters of the object using the sml model
        # name
        self.name = model.name
        # type
        # --- lets find if a tags flat, irregular, short, or long exists for this component
        for type in BoneType:
            if type in model.tags:
                self.type = str(type)
        # mesh
        self.filepath = model.mesh[0].source
        # frame
        self.frame = []
        # --- short bone
        if self.type == "short" and not len(model.mesh[0].group):
            if model.com:
                self.frame.append(model.com)
            if model.inertia:
                self.inertia.append(model.inertia)
        # --- long bone, flat bone, irregular bone
        else:
            for mesh in model.mesh:
                if len(mesh.group):
                    for tag, roi in mesh.group.iteritems():
                        # computation of mass center using the name of the ROI for the moment (This will change the ROI will be available in sml)
                        if "frameROI" in tag:
                            v_index = []
                            (v, n, t, f) = tools.loadOBJ(mesh.source)
                            for i in roi.index:
                                v_index.append(v[i])
                            # lets add it to the frame
                            frame = numpy.mean(numpy.array(v_index), 0).tolist() # position
                            frame.extend([0, 0, 0, 1]) # orientation
                            self.frame.append(frame)
                            self.inertia.append([1,0,0, 0,1,0, 0,0,1])
        # return the output
        return self

    # create the node containing affine mapped to rigid and scale
    def setup(self, parentNode, density=2000, param=None, generatedDir = None ):

        # Computation the offset according to the attribute self.transform
        offset = [0, 0, 0, 0, 0, 0, 1]
        offset[:3] = self.transform[:3]
        offset[3:] = Quaternion.from_euler(self.transform[3:6])
        scale3d = self.transform[6:]

        if self.type not in BoneType:
            self.type = "short"  # --> to set 1 frame on bone which not have a type

        # Creation of the shearless affine body
        self.body = RigidScale.API.ShearlessAffineBody(parentNode, self.name)

        # Depending on the frame set in the constructor, let decide how the body will be initialized
        if (self.frame is None) or (len(self.frame) < FramePerBoneType[self.type]):
            self.body.setFromMesh(self.filepath, density, offset, scale3d, self.voxelSize, FramePerBoneType[self.type], generatedDir=generatedDir)
            for p in self.body.frame:
                self.frame.append(p.offset())
        else:
            scale3dList = list()
            for i in range(len(self.frame)):
                scale3dList.append(scale3d)
            self.body.setManually(self.filepath, self.frame, self.voxelSize, density=1000, generatedDir=generatedDir)

        # Add of the behavior model, the collision model and the visual model
        localGeneratedDir = None if generatedDir is None else generatedDir+self.name
        self.behavior = self.body.addBehavior(self.elasticity, IntegrationPointPerBoneType[self.type], generatedDir=localGeneratedDir)
        self.collision = self.body.addCollisionMesh(self.filepath, scale3d, offset, generatedDir=localGeneratedDir)
        self.visual = self.collision.addVisualModel()
        self.visual.filename = self.collision.loader.filename
        return self.body

## ==============================================================================
## @Description: Joint to connect two bones: boneA and boneB
"""
- @param: name: bone name
- @param: boneA: access to the first bone
- @param: boneB: access to the second bone
- @param: position: joint position
- @param: mask: the free axis
- @param: limits: joint limits (on the free axis)
"""
## ==============================================================================
class Joint():

    def __init__(self, name='', boneA=None, boneB=None, frame=None, mask=[1,1,1,1,1,1], limits=[0,0,0,0,0,0]):

        # specific attributes
        self.name = name
        self.boneA = boneA
        self.boneB = boneB
        self.frame = frame
        self.mask = mask
        self.limits = limits
        self.joint = None

    def __str__(self):
        str = ""
        str =  str + "Joint: " + self.name + "\n"
        str =  str + "--- boneA -> " + self.boneA.name + "\n"
        str =  str + "--- boneB -> " + self.boneB.name + "\n"
        str =  str + "--- frame -> " + listToStr(self.frame) + "\n"
        str =  str + "--- mask -> " + listToStr(self.mask) + "\n"
        str =  str + "--- limits -> " + listToStr(self.limits)
        return str

    def initUsingSMLModel(self, model):
        self.name = model.name
        self.mask = [1]*6
        self.limits=[] # mask for limited dofs
        isLimited = False # does the joint have valid limits ?
        for d in model.dofs:
            if d.min==None or d.max==None:
                self.limits.append(0)
                self.limits.append(0)
            else:
                self.limits.append(d.min)
                self.limits.append(d.max)
            self.mask[d.index] = 0
        return self

    def setFrame(self, offset):
        self.frame = offset

    def setFramePosition(self, pos):
        self.frame[:3] = pos

    def setFrameOrientation(self, quat):
        self.frame[3:] = quat

    def setup(self, useCompliant=1, compliance=1E-6, showOffset=False, showOffsetScale=0.1):
        # Variable
        _isLimited = False
        _compliance = 0 if useCompliant else compliance
        # Needs to be fixed and used late
        boneA_offset = self.boneA.body.addAbsoluteOffset(self.boneA.name +'_offset_joint_'+self.name, self.frame)
        boneB_offset = self.boneB.body.addAbsoluteOffset(self.boneB.name +'_offset_joint_'+self.name, self.frame)
        # joint creation between the offsets
        self.joint = StructuralAPI.GenericRigidJoint(self.name, boneA_offset.node, boneB_offset.node, self.mask, _compliance)
        # add of the limits
        for l in self.limits:
            if l!=0:
                _isLimited = True
        if _isLimited:
            self.joint.addLimits(self.limits, _compliance)
        # visualization of offset: check that all frames are well placed
        boneA_offset.dofs.showObject=showOffset
        boneB_offset.dofs.showObject=showOffset
        boneA_offset.dofs.showObjectScale=showOffsetScale
        boneB_offset.dofs.showObjectScale=showOffsetScale

## ==============================================================================
## @Description: Joint to connect two bones: boneA and boneB
"""
- bone: the bone which is constraint
"""
## ==============================================================================
class Constraint(Joint):

    def __init__(self, name='', bone=None, frame=None, mask=[1,1,1,1,1,1], limits=[0,0,0,0,0,0]):
        # parent init
        Joint.__init__(self, name, bone, bone, frame, mask, limits)
        # specific attributes
        self.bone = bone

    def __str__(self):
        str = ""
        str =  str + "Constraint: " + self.name + "\n"
        str =  str + "--- bone -> " + self.bone.name + "\n"
        str =  str + "--- frame -> " + listToStr(self.frame) + "\n"
        str =  str + "--- mask -> " + listToStr(self.mask) + "\n"
        str =  str + "--- limits -> " + listToStr(self.limits)
        return str

    def initUsingSMLModel(self, model):
        # init of other parameters
        self.name = model.name
        # init of the mask and limits
        self.mask = list(straight_constraint_default_mask)
        self.limits = list(straight_constraint_default_limit)
        self.frame = list(straight_constraint_default_frame)
        # init of the offset
        self.computeFrame()
        # output
        return self

    # Set the constraint in the middle of bone, and set its orientation along the bone axe
    def computeFrame(self):
        if len(self.bone.body.frame) != 2:
            print "Only alignement constraint of the bone heads is currently handled."
            return
        if len(self.bone.frame) < len(self.bone.body.frame):
            print "You need to set some ROI to compute the bone frame for the long bone, to allow the handling of the constraint."
            return
        p1 = self.bone.frame[0]
        p2 = self.bone.frame[1]
        # Get the middel of the bone heads
        self.frame[:3] = ((numpy.array(p1[:3]) + numpy.array(p2[:3]))/2).tolist()
        # Get the right orientation
        self.frame[3:] = SofaPython.Quaternion.from_line((numpy.array(p1[:3])-numpy.array(p2[:3])).tolist())

    # Set the constraint position in the middle of the bone
    def computeFramePosition(self):
        if len(self.bone.body.frame) != 2:
            print "Only alignement constraint of the bone heads is currently handled."
            return
        p1 = self.bone.frame[0]
        p2 = self.bone.frame[1]
        # Get the middel of the bone heads
        self.frame[:3] = ((numpy.array(p1[:3]) + numpy.array(p2[:3]))/2).tolist()

    # Set the constraint orientation along the bone axe
    def computeFrameOrientation(self):
        if len(self.bone.body.frame) != 2 or len(self.bone.frame) < 2:
            print "Only alignement constraint of the bone heads is currently handled."
            return
        p1 = self.bone.frame[0]
        p2 = self.bone.frame[1]
        # Get the right orientation
        self.frame[3:] = SofaPython.Quaternion.from_line((numpy.array(p1[:3])-numpy.array(p2[:3])).tolist())

    # TO DO: fix the attributes @set and @offset de respectivement sur AssembledRigidRigidMapping, ProjectionMapping
    def setup(self, useCompliant=1, compliance=1E-6, showOffset=False, showOffsetScale=0.1):
        if len(self.bone.body.frame) != 2:
            print "Only alignement constraint of the bone heads is currently handled."
            return
        # computation of orientation
        self.computeFrameOrientation()
        # variable
        _compliance = 0 if useCompliant else compliance
        # offset computation
        bone_offset_0 = (self.bone.body.frame[0].inv()*Frame.Frame(self.frame)).offset()
        bone_offset_1 = (self.bone.body.frame[1].inv()*Frame.Frame(self.frame)).offset()
        # joint creation between the offsets
        offsetNode = self.bone.body.rigidNode.createChild(self.name+'_constraint')
        offsetNode.createObject('MechanicalObject', template='Rigid', name='DOFs', position='0 0 0 0 0 0 1  0 0 0 0 0 0 1', showObject=showOffset, showObjectScale=showOffsetScale)
        offsetNode.createObject('AssembledRigidRigidMapping', template='Rigid,Rigid', source='0 '+ listToStr(bone_offset_0) + ' 1 ' + listToStr(bone_offset_1))

        # --- joint
        jointNode = offsetNode.createChild('joint')
        jointNode.createObject('MechanicalObject', template='Vec6'+StructuralAPI.template_suffix, name='dofs')
        jointNode.createObject('RigidJointMapping', template='Rigid,Vec6'+StructuralAPI.template_suffix, name='mapping', pairs='0 1', geometricStiffness='0', translation=1, rotation=1)
        # --- constraint
        constraintNode = jointNode.createChild('constraint')
        constraintNode.createObject('MechanicalObject', template='Vec1'+StructuralAPI.template_suffix, name='dofs')
        constraintNode.createObject('MaskMapping', template='Vec6'+StructuralAPI.template_suffix+',Vec1'+StructuralAPI.template_suffix, dofs=listToStr(self.mask))
        constraintNode.createObject('UniformCompliance', template='Vec1'+StructuralAPI.template_suffix, name='compliance', isCompliance='0', compliance=_compliance)

## ==============================================================================
## @Description: Create the differents bones and register them onto a repository
"""
- @param: parentNode: the node which will contains the articulated system
- @param: model: sml model
- @param: voxel_size: voxel size used to rasterize each bone
- @param: elasticity: elasticity of each bone model
@todo: sml ids should be used as keys in the different maps, instead of names
"""
## ==============================================================================
class SceneRigidScale(SofaPython.sml.BaseScene):

    """ Builds a (sub)scene from a sml model which create a model and register all the
    re-usable information related to the frame based model"""
    def __init__(self, parentNode, model):

        SofaPython.sml.BaseScene.__init__(self, parentNode, model)

        # main components
        self.bones = dict()

        # params
        self.param.useCompliance = 0
        self.param.voxelSize = 0.005 # SI unit (m)
        self.param.elasticity = 10e3 # SI unit
        self.param.jointCompliance = 1E-6
        self.param.constraintCompliance = 1E-6

        # settings
        self.param.showRigid=False
        self.param.showRigidScale=0.05 # SI unit (m)
        self.param.showOffset=False
        self.param.showOffsetScale=0.01 # SI unit (m)
        self.param.showRigidDOFasSphere=False

        self.param.generatedDir = './model'

    def createScene(self):
        self.node.createObject('RequiredPlugin', name='image')
        self.node.createObject('RequiredPlugin', name='Flexible')
        self.node.createObject('RequiredPlugin', name='Compliant')
        self.node.createObject('RequiredPlugin', name='RigidScale')

        # bones
        SMLBones = dict()

        # rigids
        for rigidModel in self.model.getSolidsByTags({"bone"}):
            bone = Bone(rigidModel.name)
            bone.elasticity = SofaPython.units.elasticity_from_SI(self.param.elasticity)
            bone.voxelSize = SofaPython.units.length_from_SI(self.param.voxelSize)
            SMLBones[bone.name] = bone.initUsingSMLModel(rigidModel)

        # scene creation
        for b in SMLBones.values():
            self.bones[b.name] = b.setup(self.node, generatedDir=None)
            self.bones[b.name].affineDofs.showObject = self.param.showRigid
            self.bones[b.name].affineDofs.showObjectScale = SofaPython.units.length_from_SI(self.param.showRigidScale)

## ==============================================================================
## @Description: Create a sofa scene containing the articulated system
"""
- @param: parentNode: the node which will contains the articulated system
- @param: model: sml model
- @param: voxel_size: voxel size used to rasterize each bone
- @param: elasticity: elasticity of each bone model
@todo: sml ids should be used as keys in the different maps, instead of names
"""
## ==============================================================================
class SceneArticulatedRigidScale(SofaPython.sml.BaseScene):
    """ Builds a (sub)scene from a model using compliant formulation
    [tag] bone are simulated as ShearlessAffineBody
    Compliant joints are setup between the bones """
    def __init__(self, parentNode, model):

        SofaPython.sml.BaseScene.__init__(self, parentNode, model)

        # main components
        self.bones = dict()
        self.joints = dict()
        self.constraints = dict()

        # params
        self.param.useCompliance = 0
        self.param.voxelSize = 0.005 # SI unit (m)
        self.param.elasticity = 10e3 # SI unit
        self.param.jointCompliance = 1E-6
        self.param.constraintCompliance = 1E-6

        # sofa important component
        self.collisions = dict()
        self.visuals = dict()
        self.visualStyles = dict()
        self.behaviors = dict()

        # settings
        self.param.showRigid=False
        self.param.showRigidScale=0.05 # SI unit (m)
        self.param.showOffset=False
        self.param.showOffsetScale=0.01 # SI unit (m)
        self.param.showRigidDOFasSphere=False

        self.param.generatedDir = None

    def createScene(self):
        self.node.createObject('RequiredPlugin', name='image')
        self.node.createObject('RequiredPlugin', name='Flexible')
        self.node.createObject('RequiredPlugin', name='Compliant')
        self.node.createObject('RequiredPlugin', name='RigidScale')

        # settings
        if self.param.generatedDir and not os.path.exists(self.param.generatedDir):
            self.param.generatedDir = None

        # bones
        SMLBones = dict()

        # rigids
        for rigidModel in self.model.getSolidsByTags({"bone"}):
            bone = Bone(rigidModel.name)
            bone.elasticity = SofaPython.units.elasticity_from_SI(self.param.elasticity)
            bone.voxelSize = SofaPython.units.length_from_SI(self.param.voxelSize)
            SMLBones[bone.name] = bone.initUsingSMLModel(rigidModel)

        # scene creation
        for b in SMLBones.values():
            self.bones[b.name] = b.setup(self.node, generatedDir=self.param.generatedDir)
            self.bones[b.name].affineDofs.showObject = self.param.showRigid
            self.bones[b.name].affineDofs.showObjectScale = SofaPython.units.length_from_SI(self.param.showRigidScale)
            # add of behavior models
            self.behaviors[b.name] = b.behavior
            # add of collision models
            self.collisions[b.name] = b.collision
            # add of visual models
            self.visuals[b.name] = b.visual.model
            self.visualStyles[b.name] = b.visual.node.createObject('VisualStyle', displayFlags='showVisual')
            # visualisation of dofs as point
            if self.param.showRigidDOFasSphere:
                l = 0.00675 if b.type != "short" else 0.00335
                child = self.bones[b.name].rigidNode.createChild("visu")
                child.createObject('MechanicalObject', template='Vec3d', name='DOFs', showObject=1, showObjectScale=l, drawMode=4)
                child.createObject('IdentityMapping')
                b.visual.model.setColor(0.75, 0.75, 0.75, 0.5)

        # -- joints
        SMLJoint = dict() # All  the constraints
        for jointModel in self.model.genericJoints.values():
            # (self, name, boneA, boneB, frame=None, mask=[1,1,1,1,1,1], limits=[0,0,0,0,0,0]):
            boneA = None; boneB = None
            if jointModel.solids[0]:
                boneA = SMLBones[jointModel.solids[0].name]
            if jointModel.solids[1]:
                boneB = SMLBones[jointModel.solids[1].name]
            if boneA and boneB:
                frame = jointModel.offsets[0].value
                joint = Joint(jointModel.name, boneA, boneB, frame)
                SMLJoint[jointModel.name] = joint.initUsingSMLModel(jointModel)

        for j in SMLJoint.values():
            joint = j.setup(self.param.useCompliance, self.param.jointCompliance, self.param.showOffset, SofaPython.units.length_from_SI(self.param.showOffsetScale))
            self.joints[j.name] = j

        # -- constraints: only alignement constrainy will be handle for the moment
        # variables
        SMLBonesStraightConstraint = dict() # all  the constraints
        for boneModel in self.model.getSolidsByTags({"straight"}):
            bone = SMLBones[boneModel.name]
            constraint = Constraint(bone.name, bone)
            SMLBonesStraightConstraint[bone.name] = constraint.initUsingSMLModel(boneModel)

        for c in SMLBonesStraightConstraint.values():
            constraint = c.setup(self.param.useCompliance, self.param.constraintCompliance, self.param.showOffset, SofaPython.units.length_from_SI(self.param.showOffsetScale))
            self.constraints[c.name] = c