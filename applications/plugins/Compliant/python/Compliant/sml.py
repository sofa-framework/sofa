import os.path
import math
import xml.etree.ElementTree as etree

from Compliant import StructuralAPI

import SofaPython.Tools
import SofaPython.units
from SofaPython import Quaternion

from SofaPython.Tools import listToStr as concat

def parseUnits(xmlModel):
    """ set SofaPython.units.local_* to units specified in <units />
    """
    if xmlModel.find("units") is not None:
        message = "units set to:"
        xmlUnits = xmlModel.find("units")
        for unit in xmlUnits.attrib:
            exec("SofaPython.units.local_{0} = SofaPython.units.{0}_{1}".format(unit,xmlUnits.attrib[unit]))
            message+= " "+unit+":"+xmlUnits.attrib[unit]
        print message

class Mesh:
    pass

class Deformable:
    
    def __init__(self, node, name):
        self.node = node.createChild( name )
        self.dofs=None
        
    def setMesh(self, position, meshPath):
        r = Quaternion.to_euler(position[3:])  * 180.0 / math.pi
        self.meshLoader = SofaPython.Tools.meshLoader(self.node, meshPath, translation=concat(position[:3]) , rotation=concat(r))
        self.dofs = self.node.createObject("MechanicalObject", template = "Vec3d", name="dofs", src="@"+self.meshLoader.name)

def parseMesh(xmlModel):
    """ parse meshes and their attribute
    """
    meshes=dict()
    for m in xmlModel.iter("mesh"):
        if not m.find("source") is None:
            if m.attrib["id"] in meshes:
                print "WARNING: Compliant.sml.parseMesh: mesh id {0} already defined".format(m.attrib["id"])
            mesh=Mesh()
            mesh.format = m.find("source").attrib["format"]
            mesh.source = m.find("source").text
        
            mesh.group=dict()
            mesh.weight=dict()
            for g in m.iter("group"):
                mesh.group[g.attrib["id"]] =  SofaPython.Tools.strToListInt(g.find("index").text)
                mesh.weight[g.attrib["id"]] = SofaPython.Tools.strToListFloat(g.find("weight").text)
                
            meshes[m.attrib["id"]] = mesh
            
    return meshes

class Scene:
    """ Builds a (sub)scene from a sml file using compliant formulation
    
    <rigid>: if <density> is given, inertia is computed from mesh, else <mass> must be given
    """
    
    class Param:
        pass
    
    def __init__(self, filename, name=None):
        self.filename=filename
        self.name=name
        self.param=self.Param()
        # param to tune scene creation
        self.param.showRigid=False
        self.param.showRigidScale=0.5 # SI unit (m)
        self.param.showOffset=False
        self.param.showOffsetScale=0.1 # SI unit (m)
        
    def createScene(self, parentNode):
        if not os.path.exists(self.filename):
            print "ERROR: Compliant.sml.Scene: file {0} not found".format(filename)
            return None

        parentNode.createObject('RequiredPlugin', name = 'Flexible' )
        parentNode.createObject('RequiredPlugin', name = 'Compliant' )

        with open(self.filename,'r') as f:
            # relative path with respect to the scene file
            self.sceneDir = os.path.dirname(self.filename)
                        
            # TODO automatic DTD validation could go here, not available in python builtin ElementTree module
            model = etree.parse(f).getroot()
            if self.name is None:
                self.name = model.attrib["name"]
            self.node=parentNode.createChild(self.name)
            print "model:", self.name

            # units
            parseUnits(model)
            
            # meshes
            meshes = parseMesh(model)

            # rigids
            self.rigids=dict()
            for r in model.iter("rigid"):
                name = r.attrib["id"]
                if r.find("name") is not None:
                    name = r.find("name").text
                print "rigid:", name
                
                if r.attrib["id"] in self.rigids:
                    print "ERROR: Compliant.sml.scene: rigid defined twice, id:", r.attrib["id"]
                    return
                
                rigid = StructuralAPI.RigidBody(self.node, name)
                self.rigids[r.attrib["id"]] = rigid
                
                meshfile = None
                if not r.find("mesh") is None:
                    meshfile = os.path.join(self.sceneDir, meshes[r.find("mesh").attrib["id"]].source)
                offset=SofaPython.Tools.strToListFloat(r.find("position").text)

                if r.find("density") is not None:
                    rigid.setFromMesh(meshfile, density=float(r.find("density").text), offset=offset)
                else:
                    mass=1.
                    if not r.find("mass") is None:
                        mass = float(r.find("mass").text)
                    rigid.setManually(offset=offset,mass=mass)
                rigid.dofs.showObject = self.param.showRigid
                rigid.dofs.showObjectScale = SofaPython.units.length_from_SI(self.param.showRigidScale)
                # visual
                if not meshfile is None:
                    rigid.addVisualModel(meshfile)
                    rigid.addCollisionMesh(meshfile)
            
            # joints
            self.joints=dict()
            for j in model.iter("joint"):
                name = j.attrib["id"]
                if j.find("name") is not None:
                    name = j.find("name").text
                print "joint:", name
                
                if j.attrib["id"] in self.joints:
                    logging.error("ERROR: Compliant.sml.scene: joint defined twice, id:", j.attrib["id"])
                    return
                
                parent = j.find("parent")
                parentOffset = self.addOffset("offset_{0}".format(name), parent.attrib["id"], parent.find("offset"))
                child = j.find("child")
                childOffset = self.addOffset("offset_{0}".format(name), child.attrib["id"], child.find("offset"))
                
                # dofs
                mask = [1] * 6
                for dof in j.iter("dof"):
                    mask[int(dof.attrib["index"])]=0
                    #TODO limits !
                
                joint = StructuralAPI.GenericRigidJoint(name, parentOffset.node, childOffset.node, mask)
                self.joints[j.attrib["id"]] = joint
                
            #deformable
            # all rigids (bones) must be gathered in a single node
            bonesNode = self.node.createChild("bones")
            bones = bonesNode.createObject("MechanicalObject", template = "Rigid3d", name="dofs")
            input=""
            indexPairs=""
            for i,r in enumerate(self.rigids.values()):
                r.node.addChild(bonesNode)
                input += '@'+r.node.getPathName()+" "
                indexPairs += str(i) + " 0 "
                r.boneIndex=i
            bonesNode.createObject('SubsetMultiMapping', template = "Rigid3d,Rigid3d", name="mapping", input = input , output = '@./', indexPairs=indexPairs, applyRestPosition=True )
            
            self.deformable=dict()
            for d in model.iter("deformable"):
                name = d.attrib["id"]
                if d.find("name") is not None:
                    name = d.find("name").text
                print "deformable:", name
                
                position = SofaPython.Tools.strToListFloat(d.find("position").text)
                mesh = meshes[d.find("mesh").attrib["id"]]
                deformable=Deformable(self.node,name)
                bonesNode.addChild(deformable.node)
                deformable.setMesh(position, os.path.join(os.path.dirname(self.filename),mesh.source))
                
                indices=dict()
                weights=dict()
                for s in d.iter("skinning"):
                    if not s.attrib["rigid"] in self.rigids:
                        print "ERROR: Compliant.sml.Scene: skinning for deformable {0}: rigid {1} is not defined".format(name, s.attrib["rigid"])
                        continue
                    currentBone = self.rigids[s.attrib["rigid"]].boneIndex
                    if not (s.attrib["group"] in mesh.group and s.attrib["group"] in mesh.weight):
                        print "ERROR: Compliant.sml.Scene: skinning for deformable {0}: group {1} is not defined".format(name, s.attrib["group"])
                        continue
                    for index,weight in zip(mesh.group[s.attrib["group"]], mesh.weight[s.attrib["group"]]):
                        if not index in indices:
                            indices[index]=list()
                            weights[index]=list()
                        indices[index].append(currentBone)
                        weights[index].append(weight)
                
                deformable.node.createObject("LinearMapping", template="Rigid3d,Vec3d", name="skinning", input="@"+bonesNode.getPathName(), indices=concat(indices.values()), weights=concat(weights.values()))
                
                self.deformable[d.attrib["id"]]=deformable
                        
    def addOffset(self, name, rigidId, xmlOffset):
        """ add xml defined offset to rigid
        """
        if rigidId not in self.rigids:
            print "ERROR: Compliant.sml.Scene: rigid {0} is unknown".format(rigidId)
            return None
        
        if xmlOffset is None:
            # just return rigid frame
            return self.rigids[rigidId]
        
        if xmlOffset.attrib["type"] is "absolute":
            offset = self.rigids[rigidId].addAbsoluteOffset(name, SofaPython.Tools.strToListFloat(xmlOffset.text))
        else:
            offset = self.rigids[rigidId].addOffset(name, SofaPython.Tools.strToListFloat(xmlOffset.text))
        offset.dofs.showObject = self.param.showOffset
        offset.dofs.showObjectScale = SofaPython.units.length_from_SI(self.param.showOffsetScale)
        return offset
    
    
