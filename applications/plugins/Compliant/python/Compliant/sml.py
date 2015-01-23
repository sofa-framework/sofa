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
    class Group:
        pass
    pass

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

def parseData(xmlData):
    """ return the list of data in xmlData
    """
    if xmlData.attrib["type"]=="float":
        return SofaPython.Tools.strToListFloat(xmlData.text)
    elif xmlData.attrib["type"]=="int":
        return SofaPython.Tools.strToListInt(xmlData.text)
    elif xmlData.attrib["type"]=="string":
        return xmlData.text.split()

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
            mesh.data=dict()
            for g in m.iter("group"):
                mesh.group[g.attrib["id"]] = Mesh.Group()
                mesh.group[g.attrib["id"]].index = SofaPython.Tools.strToListInt(g.find("index").text)
                mesh.group[g.attrib["id"]].data = dict()
                for d in g.iter("data"):
                    mesh.group[g.attrib["id"]].data[d.attrib["name"]]=parseData(d)
                
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
        self.nodes = dict() # to store special nodes
        self.rigids = dict()
        self.joints = dict()
        self.deformable = dict()

    def insertRigids(self, modelXml, parentNode):
        for r in modelXml.iter("rigid"):
            name = r.attrib["id"]
            if r.find("name") is not None:
                name = r.find("name").text
            print "rigid:", name

            if r.attrib["id"] in self.rigids:
                print "ERROR: Compliant.sml.scene: rigid defined twice, id:", r.attrib["id"]
                return

            rigid = StructuralAPI.RigidBody(parentNode, name)
            self.rigids[r.attrib["id"]] = rigid

            meshfile = None
            if not r.find("mesh") is None:
                meshfile = os.path.join(self.sceneDir, self.meshes[r.find("mesh").attrib["id"]].source)
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

    def insertJoints(self, modelXml):
        
        for j in modelXml.iter("jointGeneric"):
            name = j.attrib["id"]
            if j.find("name") is not None:
                name = j.find("name").text
            print "joint:", name

            if j.attrib["id"] in self.joints:
                logging.error("ERROR: Compliant.sml.scene: joint defined twice, id:", j.attrib["id"])
                return

            frames=list()
            for o in j.iter("object"):
                if not o.find("offset") is None:
                    frames.append(self.addOffset("offset_{0}".format(name), o.attrib["id"], o.find("offset")))
                else:
                    frame.append(self.rigids[o.attrib["id"]])
            
            if len(frames) != 2:
                logging.error("ERROR: Compliant.sml.scene: generic joint expect two objects, {0} specified".format(len(frames)))

            # dofs
            mask = [1] * 6
            for dof in j.iter("dof"):
                mask[int(dof.attrib["index"])]=0
                #TODO limits !

            joint = StructuralAPI.GenericRigidJoint(name, frames[0].node, frames[1].node, mask)
            self.joints[j.attrib["id"]] = joint

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
            for s in d.iter("skinning"):
                if not s.attrib["rigid"] in self.rigids:
                    print "ERROR: Compliant.sml.Scene: skinning for deformable {0}: rigid {1} is not defined".format(name, s.attrib["rigid"])
                    continue
                currentBone = self.rigids[s.attrib["rigid"]].boneIndex
                if not (s.attrib["group"] in mesh.group and s.attrib["weight"] in mesh.group[s.attrib["group"]].data):
                    print "ERROR: Compliant.sml.Scene: skinning for deformable {0}: group {1} - weight {2} is not defined".format(name, s.attrib["group"], s.attrib["weight"])
                    continue
                group = mesh.group[s.attrib["group"]]
                weight = group.data[s.attrib["weight"]]
                for index,weight in zip(group.index, weight):
                    if not index in indices:
                        indices[index]=list()
                        weights[index]=list()
                    indices[index].append(currentBone)
                    weights[index].append(weight)

            if len(indices)>0:
                deformable.node.createObject("LinearMapping", template="Rigid3d,Vec3d", name="skinning", input="@"+self.nodes["bones"].getPathName(), indices=concat(indices.values()), weights=concat(weights.values()))

            self.deformable[d.attrib["id"]]=deformable

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
            modelXml = etree.parse(f).getroot()
            if self.name is None:
                self.name = modelXml.attrib["name"]
            self.node=parentNode.createChild(self.name)
            print "model:", self.name

            # units
            parseUnits(modelXml)
            
            # meshes
            self.meshes = parseMesh(modelXml)

            # rigids
            self.insertRigids(modelXml, self.node)
            
            # joints
            self.insertJoints(modelXml)

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
                        
    def addOffset(self, name, rigidId, xmlOffset):
        """ add xml defined offset to rigid
        """
        if rigidId not in self.rigids:
            print "ERROR: Compliant.sml.Scene: rigid {0} is unknown".format(rigidId)
            return None
        
        if xmlOffset is None:
            # just return rigid frame
            return self.rigids[rigidId]
        
        if xmlOffset.attrib["type"] == "absolute":
            offset = self.rigids[rigidId].addAbsoluteOffset(name, SofaPython.Tools.strToListFloat(xmlOffset.text))
        else:
            offset = self.rigids[rigidId].addOffset(name, SofaPython.Tools.strToListFloat(xmlOffset.text))
        offset.dofs.showObject = self.param.showOffset
        offset.dofs.showObjectScale = SofaPython.units.length_from_SI(self.param.showOffsetScale)
        return offset


