import os.path
import math

import SofaPython.Tools
import SofaPython.units
from SofaPython.Tools import listToStr as concat
from SofaPython import Quaternion
import Sofa

class Image:
    """ This class proposes a high-level API to build images. It support multiple meshes rasterization.
    """

    class Mesh:
        def __init__(self, value, insideValue=None):
            self.mesh=None
            self.visual=None
            self.value=value
            self.insideValue = insideValue
            self.roiValue=list() # a list of values corresponding to each roi
            self.roiIndices=None # a string pointing to indices
            self.mergeROIs=None

    def __init__(self, parentNode, name, imageType="ImageUC"):
        self.imageType = imageType
        self.node = parentNode if name=='' else parentNode.createChild(name)
        self.name = name
        self.meshes = dict()
        self.meshSeq = list() # to keep track of the mesh sequence, order does matter
        self.image = None
        self.viewer = None
        self.exporter = None


    def addMeshLoader(self, meshFile, value, insideValue=None, closingValue=None, roiIndices=list(), roiValue=list(), name=None, offset = [0,0,0,0,0,0,1], scale=[1,1,1]):
        mesh = Image.Mesh(value, insideValue)
        _name = name if not name is None else os.path.splitext(os.path.basename(meshFile))[0]
        mesh.mesh = SofaPython.Tools.meshLoader(self.node, meshFile, name="meshLoader_"+_name, triangulate=True, translation=concat(offset[:3]) , rotation=concat(Quaternion.to_euler(offset[3:])  * 180.0 / math.pi), scale3d=concat(scale))
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
            Sofa.msg_error('Image.API',"addMeshVisual : no mesh for "+ meshName)
            return
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
                Sofa.msg_error('Image.API',"addMeshToImage : no mesh for "+ name)
                return
            meshPath = SofaPython.Tools.getObjectPath(mesh.mesh)
            args["position"+str(i)]="@"+meshPath+".position"
            args["triangles"+str(i)]="@"+meshPath+".triangles"
            args["value"+str(i)]=mesh.value
            if mesh.insideValue is None:
                args["fillInside"+str(i)]=False
            else:
                args["insideValue"+str(i)]=mesh.insideValue
            if not mesh.roiIndices is None and len(mesh.roiValue)!=0 :
                args["roiIndices"+str(i)]=mesh.roiIndices
                args["roiValue"+str(i)]=SofaPython.Tools.listToStr(mesh.roiValue)
            i+=1
        self.image = self.node.createObject('MeshToImageEngine', template=self.imageType, name="image", voxelSize=voxelSize, padSize=1, subdiv=8, rotateImage=False, nbMeshes=len(self.meshes), **args)

    def addViewer(self):
        if self.image is None:
            Sofa.msg_error('Image.API',"addViewer : no image")
            return
        imagePath = SofaPython.Tools.getObjectPath(self.image)
        self.viewer = self.node.createObject('ImageViewer', name="viewer", template=self.imageType, image="@"+imagePath+".image", transform="@"+imagePath+".transform")

    def getFilename(self, filename=None, directory=""):
        _filename = filename if not filename is None else self.name+".mhd"
        _filename = os.path.join(directory, _filename)
        return _filename

    def addContainer(self, filename=None, directory="", name=''):
        self.image = self.node.createObject('ImageContainer', template=self.imageType, name='image' if name=='' else name, filename=self.getFilename(filename, directory))

    def addExporter(self, filename=None, directory=""):
        if self.image is None:
            Sofa.msg_error('Image.API',"addExporter : no image")
            return
        imagePath = SofaPython.Tools.getObjectPath(self.image)
        self.exporter = self.node.createObject('ImageExporter', template=self.imageType, name="exporter", image="@"+imagePath+".image", transform="@"+imagePath+".transform", filename=self.getFilename(filename, directory), exportAtBegin=True, printLog=True)

class Sampler:
    """ This class proposes a high-level API to build ImageSamplers
    """

    def __init__(self, parentNode, name=''):
        self.node = parentNode if name=='' else parentNode.createChild(name)
        self.name = name
        self.sampler=None
        self.mesh=None
        self.dofs=None
        self.mass=None

    def _addImageSampler(self, template, nbSamples, src, fixedPosition, **kwargs):
        self.sampler = self.node.createObject("ImageSampler", template=template, name="sampler", image=src+".image", transform=src+".transform", method="1", param=str(nbSamples)+" 1", fixedPosition=SofaPython.Tools.listListToStr(fixedPosition), **kwargs)
        return self.sampler

    def _addImageRegularSampler(self, template, src, **kwargs):
        self.sampler = self.node.createObject("ImageSampler", template=template, name="sampler", image=src+".image", transform=src+".transform", method="0", param="1", **kwargs)
        return self.sampler

    def addImageSampler(self, image, nbSamples, fixedPosition=list(), **kwargs):
        return self._addImageSampler(image.imageType, nbSamples, "@"+SofaPython.Tools.getObjectPath(image.image), fixedPosition, **kwargs)

    def addImageRegularSampler(self, image, **kwargs):
        return self._addImageRegularSampler(image.imageType, "@"+SofaPython.Tools.getObjectPath(image.image), **kwargs)

    def addMesh(self):
        if self.sampler is None:
            Sofa.msg_error('Image.API',"addMesh : no sampler")
            return None
        self.mesh = self.node.createObject('Mesh', name="mesh" ,src="@"+SofaPython.Tools.getObjectPath(self.sampler))
        return self.mesh

    def addMechanicalObject(self):
        self.dofs = self.node.createObject("MechanicalObject", template="Vec3d", name="dofs")
        return self.dofs

    def addUniformMass(self,totalMass):
        self.mass = self.node.createObject("UniformMass", template="Vec3d", name="mass", totalMass=totalMass)
        return self.mass
