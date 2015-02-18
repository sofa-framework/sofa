import SofaPython.Tools
import SofaPython.units

class Image:
    def __init__(self, parentNode, name):
        self.node = parentNode.createChild("image_"+name)
        self.name = name
        self.mesh = None
        self.value = None
        self.closingValue = None
        self.image = None
        self.viewer = None
        self.writer = None

    def addMeshLoader(self, mesh):
        self.mesh = SofaPython.Tools.meshLoader(self.node, mesh, name="loader")

    def addImage(self, voxelSize, value, closingValue=None):
        if self.mesh is None:
            print "[StructuralAPI.Image] ERROR: no mesh"
        self.value = value
        self.closingValue = value if closingValue is None else closingValue
        meshPath = SofaPython.Tools.getObjectPath(self.mesh)
        self.image = self.node.createObject('MeshToImageEngine', template="ImageUC", name="image", position="@"+meshPath+".position", triangles="@"+meshPath+".triangles", voxelSize=SofaPython.units.length_from_SI(voxelSize), value=self.value, closingValue=self.closingValue, fillInside="true", rotateImage="false",  connectivity="6") # TODO connectivity ?

    def addViewer(self):
        if self.image is None:
            print "[StructuralAPI.Image] ERROR: no image"
        imagePath = SofaPython.Tools.getObjectPath(self.image)
        self.viewer = self.node.createObject('ImageViewer', name="viewer", template="ImageUC", image="@"+imagePath+".image", transform="@"+imagePath+".transform")

    def addReader(self, filename):
        self.image = self.node.createObject('ImageContainer', template='ImageUC', name="container", filename=filename)

    def addWriter(self, filename):
        if self.image is None:
            print "[StructuralAPI.Image] ERROR: no image"
        imagePath = SofaPython.Tools.getObjectPath(self.image)
        self.node.createObject('ImageExporter', template='ImageUC', image="@"+imagePath+".image", transform="@"+imagePath+".transform", filename=filename, exportAtEnd=True, printLog=True)

