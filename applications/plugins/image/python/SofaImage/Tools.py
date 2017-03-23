import Sofa
import os.path
import SofaPython.Quaternion as quat
from SofaPython.Tools import listToStr as concat
import math
import numpy
import inspect

# be sure that the cache path exists
# automatic generation when s-t changed

class ImageProducerInfo:
    """ Used with insertImageCache
    """

    def __init__(self, imageTemplate, image, imageTransform):
        self.imageTemplate = imageTemplate
        self.image = image
        self.imageTransform = imageTransform




def insertImageCache(parentNode, imageProducerInfo, insertImageFunc, args, cachePath):
    cacheFile=os.path.join(cachePath, name+".mhd")
    if os.path.exists(cacheFilePath):
        imageObject = parentNode.createObject('ImageContainer', template=imageTemplate, name=name, filename=cacheFile, drawBB=False)
    else:
        imageObject = insertImageFunc(**args)
        assert imageObject.name == name, "image.Tools.insertImageCache: image name mismatch"
        parentNode.createObject('ImageExporter', template=imageTemplate, image="@"+image, transform="@"+transform, filename=cacheFile, exportAtEnd="1", printLog="1")
    return imageObject



def getImageTransform(filename, scaleFactor=1):
    """ Returns dim, voxelsize and rigid position of an image given an .mhd header image file
        a scaleFactor can be given to normalize image length units (usually in mm)
    """
    scale=[0,0,0]
    tr=[0,0,0]
    dim=[0,0,0]

    with open(filename,'r') as f:
        for line in f:
            splitted = line.split()
            if len(splitted)!=0:
                if 'ElementSpacing'==splitted[0] or 'spacing'==splitted[0] or 'scale3d'==splitted[0] or 'voxelSize'==splitted[0]:
                    scale = map(float,splitted[2:5])
                if 'Position'==splitted[0] or 'Offset'==splitted[0] or 'translation'==splitted[0] or 'origin'==splitted[0]:
                    tr = map(float,splitted[2:5])
                if 'Orientation'==splitted[0] or 'Rotation'==splitted[0] or 'TransformMatrix'==splitted[0] :
                    R = numpy.array([map(float,splitted[2:5]),map(float,splitted[5:8]),map(float,splitted[8:11])])
                if 'DimSize'==splitted[0] or 'dimensions'==splitted[0] or 'dim'==splitted[0]:
                    dim = map(int,splitted[2:5])
    q = quat.from_matrix(R)
    if scaleFactor!=1:
        scale = [s*scaleFactor for s in scale]
        tr = [t*scaleFactor for t in tr]
    offset=[tr[0],tr[1],tr[2],q[0],q[1],q[2],q[3]]
    return (dim,scale,offset)

def getImagePerspective(filename):
    with open(filename,'r') as f:
        for line in f:
            splitted = line.split()
            if len(splitted)!=0:
                if 'isPerpective'==splitted[0]:
                    return map(int,splitted[2:3])[0]
    return 0

def getImageType(filename):
    """ Returns type of an image given an .mhd header image file
    """
    t=""
    with open(filename,'r') as f:
        for line in f:
            splitted = line.split()
            if len(splitted)!=0:
                if 'ElementType'==splitted[0] or 'voxelType'==splitted[0] or 'scale3d'==splitted[0] or 'voxelSize'==splitted[0]:
                    t = str(splitted[2])
    if t=="MET_CHAR":
        return "ImageC"
    elif t=="MET_DOUBLE":
        return "ImageD"
    elif t=="MET_FLOAT":
        return "ImageF"
    elif t=="MET_INT":
        return "ImageI"
    elif t=="MET_LONG":
        return "ImageL"
    elif t=="MET_SHORT":
        return "ImageS"
    elif t=="MET_UCHAR":
        return "ImageUC"
    elif t=="MET_UINT":
        return "ImageUI"
    elif t=="MET_ULONG":
        return "ImageUL"
    elif t=="MET_USHORT":
        return "ImageUS"
    elif t=="MET_BOOL":
        return "ImageB"
    else:
        return None

def transformToData(scale,offset,timeOffset=0,timeScale=1,isPerspective=0):
    """ Returns a transform, formatted to sofa data given voxelsize, rigid position (offset), time and camera parameters
    """
    return concat(offset[:3])+' '+concat(quat.to_euler(offset[3:])*180./math.pi)+' '+concat(scale)+' '+str(timeOffset)+' '+str(timeScale)+' '+str(int(isPerspective))

# controller you must derived from and instanciate in the same context than your ImageViewer if you want to define actions to manually add / remove point from an image plane
class ImagePlaneController(Sofa.PythonScriptController):
    def addPoint(self, id, x, y, z):
        return

    def removePoint(self, id):
        return

    def clearPoints(self):
        return

    # return a dictionary of id -> point: {id0 : point0, idn : pointn, ...}
    # a point is defined as follows: {'position': [x, y, z], 'color': [r, g, b], ...custom parameters... }
    def getPoints(self):
        return


# simpler python script controllers based on SofaPython.script
# TODO maybe this should be double Inherited from both ImagePlaneController and SofaPython.script.Controller
# not to copy code. But then testing inheritance against ImagePlaneController has to be checked.
class Controller(ImagePlaneController):

    def __new__(cls, node, name='pythonScriptController', filename=''):
        """
        :param filename: you may have to define it (at least once) to create
                        a controller for which the class is defined in an external
                        file. Be aware the file will then be read several times.
        """

        node.createObject('PythonScriptController',
                          filename = filename,
                          classname = cls.__name__,
                          name = name)
        try:
            res = Controller.instance
            del Controller.instance
            return res
        except AttributeError:
            # if this fails, you need to call
            # Controller.onLoaded(self, node) in derived classes
            print "[SofaImage.Controller.__new__] instance not found, did you call 'SofaImage.Controller.onLoaded' on your overloaded 'onLoaded' in {} ?".format(cls)
            raise

    def onLoaded(self, node):
        Controller.instance = self


