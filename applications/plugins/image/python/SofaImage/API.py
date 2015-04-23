import os.path

import SofaPython.Tools
import SofaPython.units

class Image:
    """ This class proposes a high-level API to build images. It support multiple meshes rasterization.
    """

    class Mesh:
        def __init__(self, value, insideValue=None):
            self.mesh=None
            self.visual=None
            self.value=value
            self.insideValue = value if insideValue is None else insideValue
            self.roiValue=list() # a list of values corresponding to each roi
            self.roiIndices=None # a string pointing to indices
            self.mergeROIs=None

    def __init__(self, parentNode, name, imageType="ImageUC"):
        self.imageType = imageType
        self.node = parentNode.createChild(name)
        self.name = name
        self.meshes = dict()
        self.meshSeq = list() # to keep track of the mesh sequence, order does matter
        self.image = None
        self.viewer = None
        self.exporter = None


    def addMeshLoader(self, meshFile, value, insideValue=None, closingValue=None, roiIndices=list(), roiValue=list(), name=None):
        mesh = Image.Mesh(value, insideValue)
        _name = name if not name is None else os.path.splitext(os.path.basename(meshFile))[0]
        mesh.mesh = SofaPython.Tools.meshLoader(self.node, meshFile, name="meshLoader_"+_name, triangulate=True)
        self.__addMesh(mesh,closingValue,roiIndices,roiValue,_name)

    def addExternMesh(self, externMesh, value, insideValue=None, closingValue=None, roiIndices=list(), roiValue=list(), name=None):
        mesh = Image.Mesh(value, insideValue)
        if not name is None :
            _name = name
        else :
            _name = externMesh.name
            if "meshLoader_" in _name:
                _name=_name[len("meshLoader_"):]
        mesh.mesh = externMesh
        self.__addMesh(mesh,closingValue,roiIndices,roiValue,_name)

    def __addMesh(self, mesh, closingValue=None, roiIndices=list(), roiValue=list(), name=None):
        """ some code factorization between addMeshLoader and addExternMesh
        """
        args=dict()
        if not closingValue is None : # close mesh if closingValue is defined
            meshPath = SofaPython.Tools.getObjectPath(mesh.mesh)
            mesh.mesh = self.node.createObject("MeshClosingEngine", name="closer_"+name, inputPosition="@"+meshPath+".position", inputTriangles="@"+meshPath+".triangles")
            mesh.roiValue=[closingValue]
            mesh.roiIndices="@"+SofaPython.Tools.getObjectPath(mesh.mesh)+".indices"
        elif len(roiIndices)!=0 and len(roiValue)!=0 :
            mesh.roiValue=roiValue
            args=dict()
            for i,roi in enumerate(roiIndices):
                args["indices"+str(i+1)]=SofaPython.Tools.listToStr(roi)
            mesh.mergeROIs = self.node.createObject('MergeROIs', name="mergeROIs_"+name, nbROIs=len(roiIndices), **args)
            mesh.roiIndices="@"+SofaPython.Tools.getObjectPath(mesh.mergeROIs)+".roiIndices"
            # use mergeROIs to potentially combine other rois (from meshclosing, boxRois, etc.)
            # but here, roiIndices reformating to "[i,j,..] [k,l,..]" would work..

        self.meshes[name] = mesh
        self.meshSeq.append(name)

    def addMeshVisual(self, meshName=None, color=None):
        name = self.meshSeq[0] if meshName is None else meshName
        mesh = self.meshes[name]
        if mesh.mesh is None:
            print "[ImageAPI.Image] ERROR: no mesh for", meshName
        mesh.visual = self.node.createObject("VisualModel", name="visual_"+name, src="@"+SofaPython.Tools.getObjectPath(mesh.mesh))
        if not color is None:
            mesh.visual.setColor(color[0],color[1],color[2],color[3])

    def addAllMeshVisual(self, color=None):
        for name in self.meshSeq:
            self.addMeshVisual(name, color)

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
            args["insideValue"+(str(i) if i>1 else "")]=mesh.insideValue
            if not mesh.roiIndices is None and len(mesh.roiValue)!=0 :
                args["roiIndices"+(str(i) if i>1 else "")]=mesh.roiIndices
                args["roiValue"+(str(i) if i>1 else "")]=SofaPython.Tools.listToStr(mesh.roiValue)
            i+=1
        self.image = self.node.createObject('MeshToImageEngine', template=self.imageType, name="image", voxelSize=voxelSize, padSize="1", subdiv=8, rotateImage="false", nbMeshes=len(self.meshes), **args)

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
        self.sampler = self.node.createObject("ImageSampler", template=template, name="sampler", image=src+".image", transform=src+".transform", method="1", param=str(nbSamples)+" 1", fixedPosition=SofaPython.Tools.listListToStr(fixedPosition), **kwargs)
        return self.sampler

    #def addImageSampler(self, image, nbSamples, fixedPosition=list(), **kwargs):
        #return self._addImageSampler(nbSamples, fixedPosition, template=image.imageType, src=SofaPython.Tools.getObjectPath(image.image), **kwargs)

    def addMechanicalObject(self, template="Affine", **kwargs):
        if self.sampler is None:
            print "[ImageAPI.Sampler] ERROR: no sampler"
        samplerPath = SofaPython.Tools.getObjectPath(self.sampler)
        self.dof = self.node.createObject("MechanicalObject", template=template, name="dof", position="@"+samplerPath+".position", **kwargs)

