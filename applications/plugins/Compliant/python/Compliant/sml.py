import os.path
import xml.etree.ElementTree as etree

from Compliant import StructuralAPI

import SofaPython.Tools
import SofaPython.units

def parseUnits(xmlModel):
    """ set SofaPython.units.local_* to units specified in <units />
    """
    if not xmlModel.find("units") is None:
        message = "units set to:"
        xmlUnits = xmlModel.find("units")
        for unit in xmlUnits.attrib:
            exec("SofaPython.units.local_{0} = SofaPython.units.{0}_{1}".format(unit,xmlUnits.attrib[unit]))
            message+= " "+unit+":"+xmlUnits.attrib[unit]
        print message

class Scene:
    """ Builds a (sub)scene from a sml file using compliant formulation
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

            # rigids
            self.rigids=dict()
            for r in model.iter("rigid"):
                print "rigid:", r.attrib["name"]
                rigid = StructuralAPI.RigidBody(self.node, r.attrib["name"])
                self.rigids[r.attrib["name"]] = rigid
                # TODO set manually using <mass> if present
                rigid.setFromMesh(os.path.join(self.sceneDir, r.find("mesh").text), offset= SofaPython.Tools.strToListFloat(r.find("position").text))
                rigid.dofs.showObject = self.param.showRigid
                rigid.dofs.showObjectScale = SofaPython.units.length_from_SI(self.param.showRigidScale)
                # TODO read velocity
                # visual
                rigid.addVisualModel(r.find("mesh").text)
                rigid.addCollisionMesh(r.find("mesh").text)
            
            # joints
            self.joints=dict()
            for j in model.iter("joint"):
                print "joint:", j.attrib["name"]
                
                parent = j.find("parent")
                parentOffset = self.addOffset("offset_{0}".format(j.attrib["name"]), parent.attrib["name"], parent.find("offset"))
                child = j.find("child")
                childOffset = self.addOffset("offset_{0}".format(j.attrib["name"]), child.attrib["name"], child.find("offset"))
                
                # dofs
                mask = [1] * 6
                for dof in j.iter("dof"):
                    mask[int(dof.attrib["index"])]=0
                    #TODO limits !
                
                joint = StructuralAPI.GenericRigidJoint(self.node, j.attrib["name"], parentOffset.node, childOffset.node, mask)
                self.joints[j.attrib["name"]] = joint
                        
    def addOffset(self, name, rigidName, xmlOffset):
        """ add xml defined offset to rigid
        """
        if not rigidName in self.rigids:
            print "ERROR: Compliant.sml.Scene: rigid {0} is unknown".format(rigidName)
            return None
        
        if xmlOffset is None:
            # just return rigid frame
            return self.rigids[rigidName]
        
        if xmlOffset.attrib["type"] is "absolute":
            offset = self.rigids[rigidName].addAbsoluteOffset(name, SofaPython.Tools.strToListFloat(xmlOffset.text))
        else:
            offset = self.rigids[rigidName].addOffset(name, SofaPython.Tools.strToListFloat(xmlOffset.text))
        offset.dofs.showObject = self.param.showOffset
        offset.dofs.showObjectScale = SofaPython.units.length_from_SI(self.param.showOffsetScale)
        return offset
    
    
