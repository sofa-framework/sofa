import os.path
import math
import xml.etree.ElementTree as etree

from Compliant import StructuralAPI

import SofaPython.Tools
import SofaPython.units
from SofaPython import Quaternion
from SofaPython.Tools import listToStr as concat
import SofaPython.sml
import Flexible.sml

def insertRigid(parentNode, rigidModel, param):
    print "rigid:", rigidModel.name
    rigid = StructuralAPI.RigidBody(parentNode, rigidModel.name)
    if not rigidModel.density is None and not rigidModel.mesh is None:
        # compute physics using mesh and density
        rigid.setFromMesh(rigidModel.mesh.source, density=rigidModel.density, offset=rigidModel.position)
    elif not rigidModel.mesh is None:
        # no density but a mesh, let's compute physics whith this information plus specified mass if any
        rigid.setFromMesh(rigidModel.mesh.source, density=1, offset=rigidModel.position)
        mass=1.
        if not rigidModel.mass is None:
            mass = rigidModel.mass
        inertia = []
        for inert,m in zip(rigid.mass.inertia, rigid.mass.mass):
            for i in inert:
                inertia.append( i/m[0]*mass)
        rigid.mass.inertia = concat(inertia)
        rigid.mass.mass = mass
    else:
        # no mesh, get mass/inertia if present, default to a unit sphere
        mass=1.
        if not rigidModel.mass is None:
            mass = rigidModel.mass
        inertia = [1,1,1] #TODO: take care of full inertia matrix, which may be given in sml, update SofaPython.mass.RigidMassInfo to diagonalize it
        if not rigidModel.inertia is None:
            inertia = rigidModel.inertia
        rigid.setManually(rigidModel.position, mass, inertia)
        
    rigid.dofs.showObject = param.showRigid
    rigid.dofs.showObjectScale = SofaPython.units.length_from_SI(param.showRigidScale)
    # visual
    if not rigidModel.mesh is None:
        cm = rigid.addCollisionMesh(rigidModel.mesh.source)
        rigid.visual = cm.addVisualModel()
       
    return rigid

def insertJoint(jointModel, rigids, param):
    print "joint:", jointModel.name
    frames=list()
    for i,offset in enumerate(jointModel.offsets):
        rigid = rigids[jointModel.solids[i].id] # shortcut
        if not offset is None:
            if offset.isAbsolute():
                frames.append( rigid.addAbsoluteOffset(offset.name, offset.value))
            else:
                frames.append( rigid.addOffset(offset.name, offset.value) )
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
    
    <rigid>: if <density> is given, inertia is computed from mesh, else <mass> must be given
    """
    
    def __init__(self, parentNode, model):
        SofaPython.sml.BaseScene.__init__(self, parentNode, model)
        
        self.rigids = dict()
        self.joints = dict()
        
        self.param.showRigid=False
        self.param.showRigidScale=0.5 # SI unit (m)
        self.param.showOffset=False
        self.param.showOffsetScale=0.1 # SI unit (m)    

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
        self.nodes["armature"] = self.node.createChild("armature")
        self.nodes["armature"].createObject("MechanicalObject", template = "Rigid3d", name="dofs")
        bonesId = list() # keep track of merged bones, bone index and bone id
        input=""
        indexPairs=""
        for armatureBone in self.model.solidsByTag["armature"]:
            rigid = self.rigids[armatureBone.id]
            rigid.node.addChild(self.nodes["armature"])
            input += '@'+rigid.node.getPathName()+" "
            indexPairs += str(len(bonesId)) + " 0 "
            bonesId.append(armatureBone.id)
        self.nodes["armature"].createObject('SubsetMultiMapping', template = "Rigid3d,Rigid3d", name="mapping", input = input , output = '@./', indexPairs=indexPairs, applyRestPosition=True )
        
        #deformable
        for solidModel in self.model.solids.values():
            if len(solidModel.skinnings)>0:
                self.deformables[solidModel.id]=Flexible.sml.insertDeformableWithSkinning(self.node, solidModel, self.nodes["armature"].getPathName(), bonesId)
        
        
