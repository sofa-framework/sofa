import math

import SofaPython.Tools
from SofaPython.Tools import listToStr as concat
from SofaPython import Quaternion

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

    def addSkinning(self, bonesPath, indices, weights):
        self.skinning = self.node.createObject("LinearMapping", template="Rigid3d,Vec3d", name="skinning", input="@"+bonesPath, indices=concat(indices), weights=concat(weights))
