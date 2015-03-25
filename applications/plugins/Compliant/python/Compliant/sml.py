import os.path
import math
import xml.etree.ElementTree as etree

from Compliant import StructuralAPI

import SofaPython.Tools
import SofaPython.units
from SofaPython import Quaternion
from SofaPython.Tools import listToStr as concat
import SofaPython.mass
import SofaPython.sml
import Flexible.sml

def insertRigid(parentNode, rigidModel, param=None):
    """ create a StructuralAPI.RigidBody from the rigidModel """
    print "rigid:", rigidModel.name
    rigid = StructuralAPI.RigidBody(parentNode, rigidModel.name)

    # check mesh formats are supported by generateRigid
    meshFormatSupported = True
    for mesh in rigidModel.mesh :
        meshFormatSupported &= rigidModel.mesh.format=="obj" or rigidModel.mesh.format=="vtk"

    if len(rigidModel.mesh)!=0 and meshFormatSupported:
        massinfo = SofaPython.mass.RigidMassInfo()

        density = SofaPython.units.density_from_SI(rigidModel.density) if not rigidModel.density is None else SofaPython.units.density_from_SI(1000.)
        for mesh in rigidModel.mesh :
            mi = SofaPython.mass.RigidMassInfo()
            mi.setFromMesh(mesh.source, density=density)
            massinfo+=mi
        rigid.setFromRigidInfo(massinfo, offset=rigidModel.position , inertia_forces = False )    # TODO: handle inertia_forces ?

        if rigidModel.density is None and not rigidModel.mass is None :
            # no density but a mesh let's normalise computed mass with specified mass
            mass= SofaPython.units.mass_from_SI(rigidModel.mass)
            inertia = []
            for inert,m in zip(rigid.mass.inertia, rigid.mass.mass):
                for i in inert:
                    inertia.append( i/m[0]*mass)
            rigid.mass.inertia = concat(inertia)
            rigid.mass.mass = mass
    else:
        # no mesh, get mass/inertia if present, default to a unit sphere
        mass=SofaPython.units.mass_from_SI(1.)
        if not rigidModel.mass is None:
            mass = rigidModel.mass
        inertia = [1,1,1] #TODO: take care of full inertia matrix, which may be given in sml, update SofaPython.mass.RigidMassInfo to diagonalize it
        if not rigidModel.inertia is None:
            inertia = rigidModel.inertia
        rigid.setManually(rigidModel.position, mass, inertia)

    if not param is None:
        rigid.dofs.showObject = param.showRigid
        rigid.dofs.showObjectScale = SofaPython.units.length_from_SI(param.showRigidScale)
    # visual

    for mesh in rigidModel.mesh :
        cm = rigid.addCollisionMesh(mesh.source)
        rigid.visual = cm.addVisualModel()
       
    return rigid

def insertJoint(jointModel, rigids, param=None):
    """ create a StructuralAPI.GenericRigidJoint from the jointModel """
    print "joint:", jointModel.name
    frames=list()
    for i,offset in enumerate(jointModel.offsets):
        rigid = rigids[jointModel.solids[i].id] # shortcut
        if not offset is None:
            if offset.isAbsolute():
                frames.append( rigid.addAbsoluteOffset(offset.name, offset.value))
            else:
                frames.append( rigid.addOffset(offset.name, offset.value) )
            if not param is None:
                frames[-1].dofs.showObject = param.showOffset
                frames[-1].dofs.showObjectScale = SofaPython.units.length_from_SI(param.showOffsetScale)
        else:
            frames.append(rigid)
    mask = [1]*6
    limits=[]
    for d in jointModel.dofs:
        limits.append(d.min)
        limits.append(d.max)
        mask[d.index] = 0
    joint = StructuralAPI.GenericRigidJoint(jointModel.name, frames[0].node, frames[1].node, mask)    
    joint.addLimits(limits)
    return joint

class SceneArticulatedRigid(SofaPython.sml.BaseScene):
    """ Builds a (sub)scene from a model using compliant formulation
    [tag]rigid are simulated as RigidBody
    Compliant joints are setup between the rigids """
    
    def __init__(self, parentNode, model):
        SofaPython.sml.BaseScene.__init__(self, parentNode, model)
        
        self.rigids = dict()
        self.joints = dict()
        
        self.param.showRigid=False
        self.param.showRigidScale=0.5 # SI unit (m)
        self.param.showOffset=False
        self.param.showOffsetScale=0.1 # SI unit (m)    

    def insertMergeRigid(self, mergeNode, tag="rigid"):
        """ Merge all the rigids in a single MechanicalObject using a SubsetMultiMapping
        optionnaly give a tag to select the rigids which are merged
        return the list of merged rigids id"""

        mergeNode.createObject("MechanicalObject", template = "Rigid3d", name="dofs")
        rigidsId = list() # keep track of merged rigids, rigid index and rigid id
        input=""
        indexPairs=""
        for solid in self.model.solidsByTag[tag]:
            if not solid.id in self.rigids:
                print "[Compliant.sml.SceneArticulatedRigid.insertMergeRigid] WARNING: "+solid.name+" is not a rigid"
                continue
            rigid = self.rigids[solid.id]
            rigid.node.addChild(mergeNode)
            input += '@'+rigid.node.getPathName()+" "
            indexPairs += str(len(rigidsId)) + " 0 "
            rigidsId.append(solid.id)
        mergeNode.createObject('SubsetMultiMapping', template = "Rigid3d,Rigid3d", name="mapping", input = input , output = '@./', indexPairs=indexPairs, applyRestPosition=True )
        return rigidsId

    def createScene(self):
        self.node.createObject('RequiredPlugin', name = 'Flexible' )
        self.node.createObject('RequiredPlugin', name = 'Compliant' )
        
        SofaPython.sml.setupUnits(self.model.units)

        # rigids
        for rigidModel in self.model.solidsByTag["rigid"]:
            self.rigids[rigidModel.id] = insertRigid(self.node, rigidModel, self.param)
        
        # joints
        for jointModel in self.model.genericJoints.values():
            self.joints[jointModel.id] = insertJoint(jointModel, self.rigids, self.param)

class SceneSkinning(SceneArticulatedRigid) :
    
    def __init__(self, parentNode, model):
        SceneArticulatedRigid.__init__(self, parentNode, model)
        self.deformables = dict()
        
    def createScene(self):
        SceneArticulatedRigid.createScene(self)
        
        # all rigids (bones) must be gathered in a single node
        self.createChild(self.node, "armature")
        bonesId = self.insertMergeRigid(self.nodes["armature"], "armature")
        #deformable
        for solidModel in self.model.solids.values():
            if len(solidModel.skinnings)>0:
                self.deformables[solidModel.id]=Flexible.sml.insertDeformableWithSkinning(self.node, solidModel, self.nodes["armature"].getPathName(), bonesId)
        
        
