import Sofa

import os.path
import math
import xml.etree.ElementTree as etree 

import Quaternion
import Tools
from Tools import listToStr as concat
import units
import mass
import DAGValidation

def parseIdName(obj,objXml):
    """ set id and name of obj
    """
    obj.id = objXml.attrib["id"]
    obj.name = obj.id
    if not objXml.find("name") is None:
        obj.name = objXml.find("name").text

def parseTag(obj, objXml):
    """ set tags of the object
    """
    for xmlTag in objXml.iter("tag"):
        obj.tags.add(xmlTag.text)

def parseData(xmlData):
    """ return the list of data in xmlData
    """
    if xmlData.attrib["type"]=="float":
        return Tools.strToListFloat(xmlData.text)
    elif xmlData.attrib["type"]=="int":
        return Tools.strToListInt(xmlData.text)
    elif xmlData.attrib["type"]=="string":
        return xmlData.text.split()

class Model:
    """ This class stores a Sofa Model read from a sml/xml (Sofa Modelling Language) file.
    """

    class Mesh:

        class Group:
            def __init__(self):
                self.index=list()
                self.data=dict()
                self.tags=set()

        def __init__(self, meshXml=None):
            self.format=None
            self.source=None
            self.group=dict() # should be groups with *s*
            self.groupsByTag=dict()
            if not meshXml is None:
                self.parseXml(meshXml)

        def parseXml(self, meshXml):
            parseIdName(self,meshXml)
            self.format = meshXml.find("source").attrib["format"]
            self.source = meshXml.find("source").text
        
            for g in meshXml.iter("group"):
                self.group[g.attrib["id"]] = Model.Mesh.Group()
                self.group[g.attrib["id"]].index = Tools.strToListInt(g.find("index").text)
                for d in g.iter("data"):
                    self.group[g.attrib["id"]].data[d.attrib["name"]]=parseData(d)
                parseTag(self.group[g.attrib["id"]], g)

    class MeshAttributes:
        def __init__(self,objXml=None):
            self.collision=True
            self.simulation=True
            self.visual=True
            if not objXml is None:
                self.parseXml(objXml)

        def parseXml(self, objXml):
            if "collision" in objXml.attrib:
                self.collision = False if objXml.attrib["collision"] in {'False','0','false'} else True
            if "simulation" in objXml.attrib:
                self.simulation = False if objXml.attrib["simulation"] in {'False','0','false'} else True
            if "visual" in objXml.attrib:
                self.visual = False if objXml.attrib["visual"] in {'False','0','false'} else True

    class Solid:
        def __init__(self, solidXml=None):
            self.id = None
            self.name = None
            self.tags = set()
            self.position = None
            self.mesh = list() # list of meshes
            self.meshAttributes = dict() # attributes associated with each mesh

            #TODO replace this with a MassInfo?
            self.mass = None
            self.com = None # x,y,z
            self.inertia = None # Ixx, Ixy, Ixz, Iyy, Iyz, Izz or Ixx, Iyy, Izz
            self.inertia_rotation = None # only useful for diagonal (3 values) inertia

            self.skinnings=list()
            if not solidXml is None:
                self.parseXml(solidXml)

        def addMesh(self, mesh, attr=None):
            self.mesh.append(mesh)
            if not attr is None:
                self.meshAttributes[mesh.id]=attr
            else:
                self.meshAttributes[mesh.id]= Model.MeshAttributes()

        def parseXml(self, objXml):
            parseIdName(self, objXml)
            parseTag(self,objXml)
            self.position=Tools.strToListFloat(objXml.find("position").text)
            if not objXml.find("mass") is None:
                self.mass = float(objXml.find("mass").text)
            if not objXml.find("com") is None:
                self.com = Tools.strToListFloat(objXml.find("com").text)
            if not objXml.find("inertia") is None:
                self.inertia = Tools.strToListFloat(objXml.find("inertia").text)
            if not objXml.find("inertia_rotation") is None:
                self.inertia_rotation = Tools.strToListFloat(objXml.find("inertia_rotation").text)

    class Offset:
        def __init__(self, offsetXml=None):
            self.name = "offset"
            self.value = [0., 0., 0., 0., 0., 0., 1.] # x y z qx qy qz qw
            self.type = "absolute"
            if not offsetXml is None:
                self.parseXml(offsetXml)

        def parseXml(self, offsetXml):
            self.value = Tools.strToListFloat(offsetXml.text)
            self.type = offsetXml.attrib["type"]
            
        def isAbsolute(self):
            return self.type == "absolute"
            
    class Dof:
        def __init__(self, dofXml=None):
            self.index = None
            self.min = None
            self.max = None
            if not dofXml is None:
                self.parseXml(dofXml)

        def parseXml(self, dofXml):
            self.index = Model.dofIndex[dofXml.attrib["index"]]
            if "min" in dofXml.attrib:
                self.min = float(dofXml.attrib["min"])
            if "max" in dofXml.attrib:
                self.max = float(dofXml.attrib["max"])

    class JointGeneric:
        def __init__(self, jointXml=None):
            self.id = None
            self.name = None
            self.solids = [None,None]
            # offsets
            self.offsets = [None,None]
            # dofs
            self.dofs = []
            if not jointXml is None:
                self.parseXml(jointXml)
        
        def parseXml(self, jointXml):
            parseIdName(self,jointXml)
            solidsRef = jointXml.findall("jointSolidRef")
            for i in range(0,2):
                if not solidsRef[i].find("offset") is None:
                    self.offsets[i] = Model.Offset(solidsRef[i].find("offset"))
                    self.offsets[i].name = "offset_{0}".format(self.name)
            for dof in jointXml.iter("dof"):
                self.dofs.append(Model.Dof(dof))

    class Skinning:
        """ Skinning definition, vertices index influenced by bone with weight
        """
        def __init__(self):
            self.solid=None    # id of the parent bone
            self.mesh=None     # the target mesh
            self.index=list()  # indices of target mesh
            self.weight=list() # weights for these vertices with respect with this bone

    class Surface:
        def __init__(self):
            self.solid=None # a Model.Solid object
            self.mesh=None  # a Model.Mesh object
            self.group=None # the vertex indices of the group
            self.image=None

    class SurfaceLink:
        def __init__(self,objXml=None):
            self.id = None
            self.name = None
            self.tags = set()           # user-defined tags
            self.surfaces = [None,None] # two Model.Surface
            self.distance=None
            if not objXml is None:
                self.parseXml(objXml)

        def parseXml(self, objXml):
            parseIdName(self,objXml)
            parseTag(self,objXml)
            if objXml.find("distance"):
                self.distance=float(objXml.findText("distance"))

    dofIndex={"x":0,"y":1,"z":2,"rx":3,"ry":4,"rz":5}
    
    def __init__(self, filename=None, name=None):
        self.name=name
        self.modelDir = None
        self.units=dict()
        self.meshes=dict()
        self.solids=dict()
        self.solidsByTag=dict()
        self.surfaceLinksByTag=dict()
        self.genericJoints=dict()
        self.surfaceLinks=dict()

        if not filename is None:
            self.open(filename)

    def open(self, filename):
        self.modelDir = os.path.dirname(filename)
        with open(filename,'r') as f:
            # TODO automatic DTD validation could go here, not available in python builtin ElementTree module
            modelXml = etree.parse(f).getroot()
            if self.name is None and "name" in modelXml.attrib:
                self.name = modelXml.attrib["name"]
            else:
                self.name = os.path.basename(filename)

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
                    
            # solids
            for objXml in modelXml.iter("solid"):
                if objXml.attrib["id"] in self.solids:
                    print "ERROR: sml.Model: solid defined twice, id:", r.attrib["id"]
                    continue
                solid=Model.Solid(objXml)
                self.parseMeshes(solid, objXml)
                self.solids[solid.id]=solid

            # skinning
            for objXml in modelXml.iter("solid"):
                solid=self.solids[objXml.attrib["id"]]
                # TODO: support multiple meshes for skinning (currently only the first mesh is skinned)
                for s in objXml.iter("skinning"):
                    if not s.attrib["solid"] in self.solids:
                        print "ERROR: sml.Model: skinning for solid {0}: solid {1} is not defined".format(solid.name, s.attrib["solid"])
                        continue
                    skinning = Model.Skinning()
                    if not s.attrib["solid"] in self.solids :
                        print "ERROR: sml.Model: skinning for solid {0}: bone (solid) {1} not defined".format(solid.name, s.attrib["solid"])
                        continue
                    skinning.solid = self.solids[s.attrib["solid"]]
                    if not s.attrib["mesh"] in self.meshes :
                        print "ERROR: sml.Model: skinning for solid {0}: mesh {1} not defined".format(solid.name, s.attrib["mesh"])
                        continue
                    skinning.mesh = self.meshes[s.attrib["mesh"]]
                    #TODO: check that this mesh is also part of the solid
                    if not (s.attrib["group"] in skinning.mesh.group and s.attrib["weight"] in skinning.mesh.group[s.attrib["group"]].data):
                        print "ERROR: sml.Model: skinning for solid {0}: mesh {1} - group {2} - weight {3} is not defined".format(name, s.attrib["mesh"], s.attrib["group"], s.attrib["weight"])
                        continue
                    skinning.index = skinning.mesh.group[s.attrib["group"]].index
                    skinning.weight = skinning.mesh.group[s.attrib["group"]].data[s.attrib["weight"]]
                    solid.skinnings.append(skinning)



            # joints
            self.parseJointGenerics(modelXml)
                
            # contacts
            for c in modelXml.iter("surfaceLink"):
                if c.attrib["id"] in self.surfaceLinks:
                    print "ERROR: sml.Model: surfaceLink defined twice, id:", c.attrib["id"]
                    continue
                surfaceLink = Model.SurfaceLink(c)
                surfaces=c.findall("surface")
                for i,s in enumerate(surfaces):
                    surfaceLink.surfaces[i] = Model.Surface()
                    if s.attrib["solid"] in self.solids:
                        surfaceLink.surfaces[i].solid = self.solids[s.attrib["solid"]]
                    else:
                        print "ERROR: sml.Model: in contact {0}, unknown solid {1} referenced".format(surfaceLink.name, s.attrib["solid"])
                    if s.attrib["mesh"] in self.meshes:
                        surfaceLink.surfaces[i].mesh = self.meshes[s.attrib["mesh"]]
                    else:
                        print "ERROR: sml.Model: in contact {0}, unknown mesh {1} referenced".format(surfaceLink.name, s.attrib["mesh"])
                    if "group" in s.attrib: # optional
                        if len(s.attrib["group"]): # discard empty string
                            surfaceLink.surfaces[i].group = s.attrib["group"]
#                    if "image" in s.attrib: # optional
#                        if len(s.attrib["image"]): # discard empty string
#                            if s.attrib["image"] in self.images:
#                               reg.surfaces[i].image = self.images[s.attrib["image"]]
                self.surfaceLinks[surfaceLink.id]=surfaceLink

            self.updateTag()

    def parseUnits(self, modelXml):
        xmlUnits = modelXml.find("units")
        if not xmlUnits is None:
            for u in xmlUnits.attrib:
                self.units[u]=xmlUnits.attrib[u]
                
    def parseMeshes(self, obj, objXml):
        meshes=objXml.findall("mesh")
        for i,m in enumerate(meshes):
            meshId = m.attrib["id"]
            attr = Model.MeshAttributes(m)
            if meshId in self.meshes:
                obj.addMesh(self.meshes[meshId], attr)
            else:
                print "ERROR: sml.Model: solid {0} references undefined mesh {1}".format(obj.name, meshId)

    def parseJointGenerics(self,modelXml):
        for j in modelXml.iter("jointGeneric"):
            if j.attrib["id"] in self.genericJoints:
                print "ERROR: sml.Model: jointGeneric defined twice, id:", j.attrib["id"]
                continue

            joint=Model.JointGeneric(j)
            solids=j.findall("jointSolidRef")
            for i,o in enumerate(solids):
                if o.attrib["id"] in self.solids:
                    joint.solids[i] = self.solids[o.attrib["id"]]
                else:
                    print "ERROR: sml.Model: in joint {0}, unknown solid {1} referenced".format(joint.name, o.attrib["id"])
            self.genericJoints[joint.id]=joint

    def _setTagFromTag(self, tag, newTag, objects, objectsByTag):

        if tag in objectsByTag:
            for obj in objectsByTag[tag]:
                obj.tags.add(newTag)
        self._updateTag(objects, objectsByTag)

    def setTagFromTag(self, tag, newTag):
        """ @deprecated use setSolidTagFromTag() instead
        """
        self.setSolidTagFromTag(tag, newTag)

    def setSolidTagFromTag(self, tag, newTag):
        """ assign newTag to all solids with tag
        """
        self._setTagFromTag(tag, newTag, self.solids, self.solidsByTag)

    def setSurfaceLinkTagFromTag(self, tag, newTag):
        """ assign newTag to all surfaceLinks with tag
        """
        self._setTagFromTag(tag, newTag, self.surfaceLinks, self.surfaceLinksByTag)

    def _updateTag(self, objects, objectsByTag):
        objectsByTag.clear()
        for obj in objects.values():
            for tag in obj.tags:
                if not tag in objectsByTag:
                    objectsByTag[tag]=list()
                objectsByTag[tag].append(obj)

    def updateTag(self):
        """ Update internal Model tag structures
        Call this method after you changed solids, surfaceLinks or mesh groups tags """
        self._updateTag(self.solids, self.solidsByTag)
        self._updateTag(self.surfaceLinks, self.surfaceLinksByTag)
        for id,mesh in self.meshes.iteritems():
            self._updateTag(mesh.group, mesh.groupsByTag)

def insertVisual(parentNode, solid, color):
    node = parentNode.createChild("node_"+solid.name)
    translation=solid.position[:3]
    rotation = Quaternion.to_euler(solid.position[3:])  * 180.0 / math.pi
    for m in solid.mesh:
        Tools.meshLoader(node, m.source, name="loader_"+solid.name)
        node.createObject("OglModel",src="@loader_"+solid.name, translation=concat(translation),rotation=concat(rotation), color=color)
    
def setupUnits(myUnits):
    message = "units:"
    for quantity,unit in myUnits.iteritems():
        exec("units.local_{0} = units.{0}_{1}".format(quantity,unit))
        message+=" "+quantity+":"+unit
    print message    

def getSolidRigidMassInfo(solid, density):
    massInfo = mass.RigidMassInfo()
    for mesh in solid.mesh:
        if solid.meshAttributes[mesh.id].simulation is True:
            # mesh mass info
            mmi = mass.RigidMassInfo()
            mmi.setFromMesh(mesh.source, density=density)
            massInfo += mmi
    return massInfo

class BaseScene:
    """ Base class for Scene class, creates a node for this Scene
    """
    class Param:
        pass

    def __init__(self,parentNode,model,name=None):
        self.root = parentNode
        self.model = model
        self.param = BaseScene.Param()
        self.material = Tools.Material() # a default material set
        self.solidMaterial = dict() # assign a material to a solid
        self.nodes = dict() # to store special nodes
        n=name
        if n is None:
            n=self.model.name
        self.node=parentNode.createChild(self.model.name)

    def createChild(self, parent, childName):
        """Creates a child node and store it in the Scene nodes dictionary"""
        """ if parent is a list of Nodes, the child is created in the fist valid parent """
        """ and then added to every other valid parents """
        childNode = None
        if isinstance(parent, list): # we have a list of parent nodes
            for p in parent:
                if not p is None: # p is valid
                    if childNode is None: # childNode is not yet created
                        childNode = p.createChild( childName )
                    else:
                        p.addChild( childNode )
        else: # only one parent
            childNode = parent.createChild(childName)
        self.nodes[childName] = childNode
        return childNode
        
    def setMaterial(self, solid, material):
        """ assign material to solid
        """
        self.solidMaterial[solid]=material
    
    def setMaterialByTag(self, tag, material):
        """ assign material to all solids with tag
        """
        if tag in self.model.solidsByTag:
            for solid in self.model.solidsByTag[tag]:
                self.solidMaterial[solid.id] = material
    
    def getMaterial(self, solid):
        """ return the solid material, "default" if none is defined
        """
        if solid in self.solidMaterial:
            return self.solidMaterial[solid]
        else :
            return "default"

    def getCollision(self,solidId,meshId):
        """ returns a collision object identified by solidId/meshId
        """
        mesh=None
        if hasattr(self, 'rigids'):  # inserted by Compliant.sml
            if solidId in self.rigids:
                if meshId in self.rigids[solidId].collisions:
                    mesh = self.rigids[solidId].collisions[meshId]
        if hasattr(self, 'collisions'):  # inserted by Anatomy.sml
            if solidId in self.collisions:
                if meshId in self.collisions[solidId]:
                    mesh = self.collisions[solidId][meshId]
        return mesh

    def getVisual(self,solidId,meshId):
        """ returns a visual object identified by solidId/meshId
        """
        mesh=None
        if hasattr(self, 'rigids'):  # inserted by Compliant.sml
            if solidId in self.rigids:
                if meshId in self.rigids[solidId].visuals:
                    mesh = self.rigids[solidId].visuals[meshId]
        if hasattr(self, 'visuals'):  # inserted by Anatomy.sml
            if solidId in self.visuals:
                if meshId in self.visuals[solidId]:
                    mesh = self.visuals[solidId][meshId]
        return mesh

    def dagValidation(self):
        err = DAGValidation.test( self.root, True )
        if not len(err) is 0:
            print "ERROR (SofaPython.BaseScene) your DAG scene is not valid"
            for e in err:
                print e

class SceneDisplay(BaseScene):
    """ Creates a scene to display solid meshes
    """
    def __init__(self,parentNode,model):
        BaseScene.__init__(self,parentNode,model)
        self.param.colorDefault="1. 1. 1."
        self.param.colorByTag=dict()
   
    def getTagColor(self, tags):
        """ get the color from the given tags
        if several tags are defined, which corresponds to several colors, one of these color is returned
        """
        tag = tags & set(self.param.colorByTag.keys())
        if len(tag)==0:
            return self.param.colorDefault
        else:
            return self.param.colorByTag[tag.pop()]


    def createScene(self):
        model=self.model # shortcut
        for solid in model.solids.values():
            print "Display solid:", solid.name
            color = self.getTagColor(solid.tags)
            insertVisual(self.node, solid, color)
