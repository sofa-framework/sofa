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
    if not rigidModel.density is None:
        rigid.setFromMesh(rigidModel.mesh.source, density=rigidModel.density, offset=rigidModel.position)
    else:
        mass=1.
        if not rigidModel.mass is None:
            mass = rigidModel.mass
        rigid.setManually(offset=rigidModel.position,mass=mass)
    rigid.dofs.showObject = param.showRigid
    rigid.dofs.showObjectScale = SofaPython.units.length_from_SI(param.showRigidScale)
    # visual
    if not rigidModel.mesh is None:
        rigid.addVisualModel(rigidModel.mesh.source)
        rigid.addCollisionMesh(rigidModel.mesh.source)
    return rigid

def insertJoint(jointModel, rigids, param):
    print "joint:", jointModel.name
    frames=list()
    for i,offset in enumerate(jointModel.offsets):
        rigid = rigids[jointModel.objects[i].id] # shortcut
        if not offset is None:
            if offset.isAbsolute():
                frames.append( rigid.addAbsoluteOffset(offset.name, offset.value))
            else:
                frames.append( rigid.addOffset(offset.name, offset.value) )
            frames[-1].dofs.showObject = param.showOffset
            frames[-1].dofs.showObjectScale = SofaPython.units.length_from_SI(param.showOffsetScale)
        else:
            frames.append(rigid)
    mask = [(1-d) for d in jointModel.dofs]
    joint = StructuralAPI.GenericRigidJoint(jointModel.name, frames[0].node, frames[1].node, mask) 
    #TODO limits !
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
        for rigidId,rigidModel in self.model.rigids.iteritems():
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
        self.nodes["bones"] = self.node.createChild("bones")
        self.nodes["bones"].createObject("MechanicalObject", template = "Rigid3d", name="dofs")
        bonesId = list() # keep track of merged bones, bone index and bone id
        input=""
        indexPairs=""
        for rigidId,rigid in self.rigids.iteritems():
            rigid.node.addChild(self.nodes["bones"])
            input += '@'+rigid.node.getPathName()+" "
            indexPairs += str(len(bonesId)) + " 0 "
            bonesId.append(rigidId)
        self.nodes["bones"].createObject('SubsetMultiMapping', template = "Rigid3d,Rigid3d", name="mapping", input = input , output = '@./', indexPairs=indexPairs, applyRestPosition=True )
        
        #deformable
        for deformableModel in self.model.deformables.values():
            self.deformables[deformableModel.id]=Flexible.sml.insertDeformableWithSkinning(self.node, deformableModel, self.nodes["bones"].getPathName(), bonesId)
        
        