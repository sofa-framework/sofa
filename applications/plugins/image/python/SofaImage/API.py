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
        self.prefix = ""
        self.node = parentNode if name=='' else parentNode.createChild(name)
        self.name = name
        self.meshes = dict()
        self.meshSeq = list() # to keep track of the mesh sequence, order does matter
        self.image = None
        self.viewer = None
        self.exporter = None

    def getImagePath(self):
        return self.image.getLinkPath()

    def template(self):
        return self.prefix+self.imageType

    def addContainer(self, filename=None, directory="", name=''):
        self.image = self.node.createObject('ImageContainer', template=self.template(), name='image' if name=='' else name, filename=self.getFilename(filename, directory))

    def fromTransferFunction(self, inputImage, param):
        """ create by applying TransferFunction to input image
            note: using a different name than container allows inplace use of this function
        """
        self.node.createObject("TransferFunction", template=inputImage.template()+","+self.template(), name="transferFunction", inputImage=inputImage.getImagePath()+".branchingImage", param=param)
        self.image = self.node.createObject("ImageContainer", template=self.template(), name="transferedImage", branchingImage="@transferFunction.outputImage", transform=inputImage.getImagePath()+".transform", drawBB="false")

    def fromImages(self, images,  overlap="1"):
        """ Merge the input regular images using the MergeImagesIntoBranching component, returns the corresponding BranchingImage.API.Image
        """
        args=dict()
        i=1
        for im in images:
            args["image"+str(i)]=im.getImagePath()+".image"
            args["transform"+str(i)]=im.getImagePath()+".transform"
            i+=1
        self.image = self.node.createObject("MergeImages", template=images[0].template()+","+self.template(), name="mergedImage", nbImages=len(images), overlap=overlap, **args )
        for parentImage in images:
            parentImage.node.addChild(self.node)


    def addExporter(self, filename=None, directory=""):
        self.exporter = self.node.createObject('ImageExporter', template=self.template(), name="exporter", image=self.getImagePath()+".image", transform=self.getImagePath()+".transform", filename=self.getFilename(filename, directory), exportAtBegin=True, printLog=True)

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
            meshPath = mesh.mesh.getLinkPath()
            mesh.mesh = self.node.createObject("MeshClosingEngine", name="closer_"+name, inputPosition=meshPath+".position", inputTriangles=meshPath+".triangles")
            mesh.roiValue=[closingValue]
            mesh.roiIndices=mesh.mesh.getLinkPath()+".indices"
        elif len(roiIndices)!=0 and len(roiValue)!=0 :
            mesh.roiValue=roiValue
            args=dict()
            for i,roi in enumerate(roiIndices):
                args["indices"+str(i+1)]=concat(roi)
            mesh.mergeROIs = self.node.createObject('MergeROIs', name="mergeROIs_"+name, nbROIs=len(roiIndices), **args)
            mesh.roiIndices=mesh.mergeROIs.getLinkPath()+".roiIndices"
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
        mesh.visual = self.node.createObject("VisualModel", name="visual_"+name, src=mesh.mesh.getLinkPath())
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
            meshPath = mesh.mesh.getLinkPath()
            args["position"+str(i)]=meshPath+".position"
            args["triangles"+str(i)]=meshPath+".triangles"
            args["value"+str(i)]=mesh.value
            if mesh.insideValue is None:
                args["fillInside"+str(i)]=False
            else:
                args["insideValue"+str(i)]=mesh.insideValue
            if not mesh.roiIndices is None and len(mesh.roiValue)!=0 :
                args["roiIndices"+str(i)]=mesh.roiIndices
                args["roiValue"+str(i)]=concat(mesh.roiValue)
            i+=1
        self.image = self.node.createObject('MeshToImageEngine', template=self.template(), name="image", voxelSize=voxelSize, padSize=1, subdiv=8, rotateImage=False, nbMeshes=len(self.meshes), **args)

    def addViewer(self):
        if self.image is None:
            Sofa.msg_error('Image.API',"addViewer : no image")
            return
        self.viewer = self.node.createObject('ImageViewer', name="viewer", template=self.imageType, image=self.getImagePath()+".image", transform=self.getImagePath()+".transform")

    def getFilename(self, filename=None, directory=""):
        _filename = filename if not filename is None else self.name+".mhd"
        _filename = os.path.join(directory, _filename)
        return _filename


    def createTransferFunction(self, parentNode, name, param, imageType="ImageR"):
        """ Apply a TransferFunction component applied to this image
        Create an output image in parentNode to store the the result
        returns the corresponding SofaImage.API.Image
        """
        inputImagePath = SofaPython.Tools.getObjectPath(self.image)
        outputImage = Image(parentNode, name, imageType=imageType)
        outputImage.node.createObject("TransferFunction", template=self.imageType+","+outputImage.imageType, name="transferFunction", inputImage="@"+inputImagePath+".image", param=param)
        outputImage.image = outputImage.node.createObject("ImageContainer", template=outputImage.imageType, name="image", image="@transferFunction.outputImage", transform="@"+inputImagePath+".transform", drawBB="false")
        return outputImage

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

    def addImageSampler(self, image, nbSamples, fixedPosition=list(), **kwargs):
        self.sampler = self.node.createObject("ImageSampler", template=image.template(), name="sampler", image=image.getImagePath()+".image", transform=image.getImagePath()+".transform", method="1", param=str(nbSamples)+" 1", fixedPosition=SofaPython.Tools.listListToStr(fixedPosition), **kwargs)

    def addImageRegularSampler(self, image, **kwargs):
        self.sampler = self.node.createObject("ImageSampler", template=image.template(), name="sampler", image=image.getImagePath()+".image", transform=image.getImagePath()+".transform", method="0", param="1", **kwargs)

    def addMesh(self):
        if self.sampler is None:
            Sofa.msg_error('Image.API',"addMesh : no sampler")
            return None
        self.mesh = self.node.createObject('Mesh', name="mesh" ,src=self.sampler.getLinkPath())
        return self.mesh

    def addMechanicalObject(self, template="Vec3"):
        if self.sampler is None:
            Sofa.msg_error('Image.API',"addMechanicalObject : no sampler")
            return None
        if self.mesh is None:
            self.dofs = self.node.createObject("MechanicalObject", template=template, name="dofs", position='@sampler.position')
        else:
            self.dofs = self.node.createObject("MechanicalObject", template=template, name="dofs")
        return self.dofs

    def addUniformMass(self,totalMass):
        self.mass = self.node.createObject("UniformMass", template="Vec3", name="mass", totalMass=totalMass)
        return self.mass
