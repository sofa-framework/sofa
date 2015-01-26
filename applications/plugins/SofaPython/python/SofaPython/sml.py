import Sofa

import os.path
import math
import xml.etree.ElementTree as etree 

import Quaternion
import Tools
from Tools import listToStr as concat
import units

def parseIdName(obj,objXml):
    """ set id and name of obj
    """
    obj.id = objXml.attrib["id"]
    obj.name = obj.id
    if not objXml.find("name") is None:
        obj.name = objXml.find("name").text

class Model:

    class Mesh:

        class Group:
            pass
        
        def __init__(self, meshXml):
            self.format = meshXml.find("source").attrib["format"]
            self.source = meshXml.find("source").text
        
            self.group=dict()
            self.data=dict()
            for g in meshXml.iter("group"):
                self.group[g.attrib["id"]] = Mesh.Group()
                self.group[g.attrib["id"]].index = SofaPython.Tools.strToListInt(g.find("index").text)
                self.group[g.attrib["id"]].data = dict()
                for d in g.iter("data"):
                    self.group[g.attrib["id"]].data[d.attrib["name"]]=parseData(d)                    

    
    class Rigid:
        def __init__(self, xml):
            parseIdName(self,xml)
            self.position=Tools.strToListFloat(xml.find("position").text)
            if not xml.find("mesh") is None:
                self.mesh = xml.find("mesh").attrib["id"]
            if not xml.find("density") is None:
                self.density=float(xml.find("density").text)
            if not xml.find("mass") is None:
                self.mass = float(xml.find("mass").text)
            
    #class JointGeneric:
        #def __init__(self, name="Unknown",object1,offset1,object2,offset2):
        #pass
    class Deformable:
        def __init__(self,xml):
            parseIdName(self,xml)
            self.position = Tools.strToListFloat(xml.find("position").text)
            self.mesh = xml.find("mesh").attrib["id"]
            self.indices=dict()
            self.weights=dict()
    
    dofIndex={"x":0,"y":1,"z":2,"rx":3,"ry":4,"rz":5}
    
    def __init__(self, filename, name=None):
        self.name=os.path.basename(filename)
        self.modelDir = os.path.dirname(filename)
        self.units=dict()
        self.meshes=dict()
        self.rigids=dict()
        #self.rigidsbyType=dict()
        self.jointGenerics=dict()
        self.deformables=dict()
        #self.deformablesByType=dict()
        
        with open(filename,'r') as f:
            # TODO automatic DTD validation could go here, not available in python builtin ElementTree module
            modelXml = etree.parse(f).getroot()
            if name is None:
                self.name = modelXml.attrib["name"]

            # units
            self.parseUnits(modelXml)
            # meshes
            for m in modelXml.iter("mesh"):
                if not m.find("source") is None:
                    if m.attrib["id"] in self.meshes:
                        print "WARNING: sml.Model: mesh id {0} already defined".format(m.attrib["id"])
                    mesh = Model.Mesh(m)
                    sourceFullPath = os.path.join(self.modelDir,mesh.source)
                    if os.path.exists(sourceFullPath):
                        mesh.source=sourceFullPath
                    else:
                        print "WARNING: sml.Model: mesh not found:", mesh.source
                    self.meshes[m.attrib["id"]] = mesh
                    
            # rigids
            for r in modelXml.iter("rigid"):
                if r.attrib["id"] in self.rigids:
                    print "ERROR: sml.Model: rigid defined twice, id:", r.attrib["id"]
                    continue
                rigid=Model.Rigid(r)
                self.rigids[rigid.id]=rigid
            
            # joints
            #self.parseJoints(modelXml)
            
            #deformable
            for d in modelXml.iter("deformable"):
                if d.attrib["id"] in self.deformables:
                    print "ERROR: sml.Model: deformable defined twice, id:", d.attrib["id"]
                    continue
                deformable=Model.Deformable(d)
                mesh=self.meshes[deformable.mesh] # shortcut
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
                        deformable.indices[index].append(currentBone)
                        deformable.weights[index].append(weight)
                self.deformables[deformable.id]=deformable

    def parseUnits(self, modelXml):
        xmlUnits = modelXml.find("units")
        if not xmlUnits is None:
            for u in xmlUnits.attrib:
                self.units[u]=float(xmlUnits.attrib[u])

    #def parseJointGenerics(self,modelXml): 
        #for j in modelXml.iter("jointGeneric"):
            #joint=JointGeneric()
            #parseIdName(joint,j)

            #if j.attrib["id"] in self.joints:
                #print "ERROR: sml.Model: joint defined twice, id:", j.attrib["id"]
                #continue

            #frames=list()
            #for o in j.iter("object"):
                #if not o.find("offset") is None:
                    #frames.append(self.addOffset("offset_{0}".format(name), o.attrib["id"], o.find("offset")))
                #else:
                    #frames.append(self.rigids[o.attrib["id"]])
            
            #if len(frames) != 2:
                #logging.error("ERROR: Compliant.sml.scene: generic joint expect two objects, {0} specified".format(len(frames)))

            ## dofs
            #mask = [1] * 6
            #for dof in j.iter("dof"):
                #mask[dofIndex[dof.attrib["index"]]]=0
                ##TODO limits !

            #self.jointGeneric[r.attrib["id"]]=rigid

            #joint = StructuralAPI.GenericRigidJoint(name, frames[0].node, frames[1].node, mask)
            #self.jointGenerics[j.attrib["id"]] = joint
            

class BaseScene:
    class Param:
        pass
    def __init__(self,parentNode,model):
        self.model=model
        self.param=BaseScene.Param()
        self.node=parentNode.createChild(self.model.name)
        
    def setupUnits(self):
        message = "units:"
        for unit,value in self.model.units.iteritems():
            exec("SofaPython.units.local_{0} = SofaPython.units.{0}_{1}".format(unit,value))
            message+=" "+unit+":"+value
        print message

class SceneDisplay(BaseScene):
    def __init__(self,parentNode,model):
        BaseScene.__init__(self,parentNode,model)
        self.param.rigidColor="1. 0. 0."
        self.param.deformableColor="0. 1. 0."

    def insertVisual(self,name,mesh,position,color):
        node = self.node.createChild("node_"+name)
        print "position:",position
        translation=position[:3]
        rotation = Quaternion.to_euler(position[3:])  * 180.0 / math.pi
        Tools.meshLoader(node, mesh, name="loader_"+name, translation=concat(translation),rotation=concat(rotation))
        node.createObject("OglModel",src="@loader_"+name, color=color)
        
    def createScene(self):
        model=self.model # shortcut
        for name,rigid in model.rigids.iteritems():
            print "Display rigid:",name
            self.insertVisual(name,model.meshes[rigid.mesh].source,rigid.position,self.param.rigidColor)
        
        for name,deformable in model.deformables.iteritems():
            print "Display deformable:",name
            self.insertVisual(name,model.meshes[deformable.mesh].source,deformable.position,self.param.deformableColor)
            