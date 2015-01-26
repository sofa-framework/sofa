import os.path
import math
import xml.etree.ElementTree as etree

from Compliant import StructuralAPI

import SofaPython.Tools
import SofaPython.units
from SofaPython import Quaternion
from SofaPython.Tools import listToStr as concat
import SofaPython.sml



class Deformable:
    
    def __init__(self, node, name):
        self.node = node.createChild( name )
        self.dofs=None
        
    def setMesh(self, position, meshPath):
        r = Quaternion.to_euler(position[3:])  * 180.0 / math.pi
        self.meshLoader = SofaPython.Tools.meshLoader(self.node, meshPath, translation=concat(position[:3]) , rotation=concat(r))
        self.topology = self.node.createObject('MeshTopology', name='topology', src="@"+self.meshLoader.name )
        self.dofs = self.node.createObject("MechanicalObject", template = "Vec3d", name="dofs", src="@"+self.meshLoader.name)
        
    def addVisual(self):
        return Deformable.VisualModel(self.node)
    
    class VisualModel:
        def __init__(self, node ):
            self.node = node.createChild("visual")
            self.model = self.node.createObject('VisualModel', name="model")
            self.mapping = self.node.createObject('IdentityMapping', name="mapping")

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

class Scene(SofaPython.sml.BaseScene):
    """ Builds a (sub)scene from a model using compliant formulation
    
    <rigid>: if <density> is given, inertia is computed from mesh, else <mass> must be given
    """
    
    def __init__(self, parentNode, model):
        SofaPython.sml.BaseScene.__init__(self, parentNode, model)
        
        self.rigids = dict()
        self.joints = dict()
        self.deformable = dict()
        
        self.param.showRigid=False
        self.param.showRigidScale=0.5 # SI unit (m)
        self.param.showOffset=False
        self.param.showOffsetScale=0.1 # SI unit (m)    

    def insertDeformables(self, modelXml, parentNode):
        for d in modelXml.iter("deformable"):
            name = d.attrib["id"]
            if d.find("name") is not None:
                name = d.find("name").text
            print "deformable:", name

            position = SofaPython.Tools.strToListFloat(d.find("position").text)
            mesh = self.meshes[d.find("mesh").attrib["id"]]
            deformable=Deformable(parentNode,name)
            self.nodes["bones"].addChild(deformable.node)
            deformable.setMesh(position, os.path.join(os.path.dirname(self.filename),mesh.source))
            deformable.addVisual()

            indices=dict()
            weights=dict()
            

            if len(indices)>0:
                deformable.node.createObject("LinearMapping", template="Rigid3d,Vec3d", name="skinning", input="@"+self.nodes["bones"].getPathName(), indices=concat(indices.values()), weights=concat(weights.values()))

            self.deformable[d.attrib["id"]]=deformable

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
        
        return

        # all rigids (bones) must be gathered in a single node
        self.nodes["bones"] = self.node.createChild("bones")
        bones = self.nodes["bones"].createObject("MechanicalObject", template = "Rigid3d", name="dofs")
        input=""
        indexPairs=""
        for i,r in enumerate(self.rigids.values()):
            r.node.addChild(self.nodes["bones"])
            input += '@'+r.node.getPathName()+" "
            indexPairs += str(i) + " 0 "
            r.boneIndex=i
        self.nodes["bones"].createObject('SubsetMultiMapping', template = "Rigid3d,Rigid3d", name="mapping", input = input , output = '@./', indexPairs=indexPairs, applyRestPosition=True )
        
        #deformable
        self.insertDeformables(modelXml, parentNode)
                        
