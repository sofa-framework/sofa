import os.path

import SofaPython.Tools
import SofaPython.units

class Image:
    """ This class proposes a high-level API to build images. It support multiple meshes rasterization.
    """

    class Mesh:
        # TODO add support for ROI
        def __init__(self, value, closingValue=None):
            self.mesh=None
            self.visual=None
            self.value=value
            self.closingValue = value if closingValue is None else closingValue

    def __init__(self, parentNode, name, imageType="ImageUC"):
        self.imageType = imageType
        self.node = parentNode.createChild(name)
        self.name = name
        self.meshes = dict()
        self.meshSeq = list() # to keep track of the mesh sequence, order does matter
        self.value = None
        self.closingValue = None
        self.image = None
        self.viewer = None
        self.exporter = None

    def addMeshLoader(self, meshFile, value, closingValue=None, name=None):
        mesh = Image.Mesh(value, closingValue)
        _name = name if not name is None else os.path.splitext(os.path.basename(meshFile))[0]
        mesh.mesh = SofaPython.Tools.meshLoader(self.node, meshFile, name="meshLoader_"+_name, triangulate=True)
        self.meshes[_name] = mesh
        self.meshSeq.append(_name)

    def addMesh(self, externMesh, value, closingValue=None, name=None):
        mesh = Image.Mesh(value, closingValue)
        if not name is None :
            _name = name
        else :
            _name = externMesh.name
            if "meshLoader_" in _name:
                _name=_name[len("meshLoader_"):]
        mesh.mesh = externMesh
        self.meshes[_name] = mesh
        self.meshSeq.append(_name)

    def addMeshVisual(self, meshName=None):
        name = self.meshSeq[0] if meshName is None else meshName
        mesh = self.meshes[name]

        if mesh.mesh is None:
            print "[ImageAPI.Image] ERROR: no mesh for", meshName
        mesh.visual = self.node.createObject("VisualModel", name="visual_"+name, src="@"+SofaPython.Tools.getObjectPath(mesh.mesh))

    def addAllMeshVisual(self, color=None):
        for name in self.meshSeq:
            self.addMeshVisual(name)
            if not color is None:
                self.meshes[name].visual.setColor(color[0],color[1],color[2],color[3])

    def addMeshToImage(self, voxelSize):
        args=dict()
        i=1
        for name in self.meshSeq:
            mesh = self.meshes[name]
            if mesh.mesh is None:
                print "[ImageAPI.Image] ERROR: no mesh for", name
            meshPath = SofaPython.Tools.getObjectPath(mesh.mesh)
            args["position"+(str(i) if i>1 else "")]="@"+meshPath+".position"
            args["triangles"+(str(i) if i>1 else "")]="@"+meshPath+".triangles"
            args["value"+(str(i) if i>1 else "")]=mesh.value
#            args["closingValue"+(str(i) if i>1 else "")]=mesh.closingValue not yet a closingValue per mesh !
            args["closingValue"]=mesh.closingValue # last wins
            i+=1
        self.image = self.node.createObject('MeshToImageEngine', template=self.imageType, name="image", voxelSize=SofaPython.units.length_from_SI(voxelSize), subdiv=8, fillInside="true", rotateImage="false", nbMeshes=len(self.meshes), **args)

    def addViewer(self):
        if self.image is None:
            print "[ImageAPI.Image] ERROR: no image"
        imagePath = SofaPython.Tools.getObjectPath(self.image)
        self.viewer = self.node.createObject('ImageViewer', name="viewer", template=self.imageType, image="@"+imagePath+".image", transform="@"+imagePath+".transform")

    def getFilename(self, filename=None, directory=""):
        _filename = filename if not filename is None else self.name+".mhd"
        _filename = os.path.join(directory, _filename)
        return _filename

    def addContainer(self, filename=None, directory=""):
        self.image = self.node.createObject('ImageContainer', template=self.imageType, name="image", filename=self.getFilename(filename, directory))

    def addExporter(self, filename=None, directory=""):
        if self.image is None:
            print "[ImageAPI.Image] ERROR: no image"
        imagePath = SofaPython.Tools.getObjectPath(self.image)
        self.exporter = self.node.createObject('ImageExporter', template=self.imageType, name="exporter", image="@"+imagePath+".image", transform="@"+imagePath+".transform", filename=self.getFilename(filename, directory), exportAtEnd=True, printLog=True)

class Sampler:
    """ This class proposes a high-level API to build ImageSamplers
    """

    def __init__(self, parentNode, name):
        self.node = parentNode.createChild(name)
        self.name = name
        self.sampler=None
        self.dof=None

    def _addImageSampler(self, template, nbSamples, src, fixedPosition, **kwargs):
        self.sampler = self.node.createObject("ImageSampler", template=template, name="sampler", src=src, method="1", param=str(nbSamples)+" 1", fixedPosition=SofaPython.Tools.listListToStr(fixedPosition), **kwargs)
        return self.sampler

    #def addImageSampler(self, image, nbSamples, fixedPosition=list(), **kwargs):
        #return self._addImageSampler(nbSamples, fixedPosition, template=image.imageType, src=SofaPython.Tools.getObjectPath(image.image), **kwargs)

    def addMechanicalObject(self, template="Affine", **kwargs):
        if self.sampler is None:
            print "[ImageAPI.Sampler] ERROR: no sampler"
        samplerPath = SofaPython.Tools.getObjectPath(self.sampler)
        self.dof = self.node.createObject("MechanicalObject", template=template, name="dof", position="@"+samplerPath+".position", **kwargs)

