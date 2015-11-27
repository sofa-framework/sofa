import os.path
import SofaPython.Quaternion as quat
from SofaPython.Tools import listToStr as concat
import math
import numpy

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
            if 'ElementSpacing'==splitted[0] or 'spacing'==splitted[0] or 'scale3d'==splitted[0] or 'voxelSize'==splitted[0]:
                scale = map(float,splitted[2:5])
            if 'Position'==splitted[0] or 'Offset'==splitted[0] or 'translation'==splitted[0] or 'origin'==splitted[0]:
                tr = map(float,splitted[2:5])
            if 'Orientation'==splitted[0] :
                R = numpy.array([map(float,splitted[2:5]),map(float,splitted[5:8]),map(float,splitted[8:11])])
            if 'DimSize'==splitted[0] or 'dimensions'==splitted[0] or 'dim'==splitted[0]:
                dim = map(int,splitted[2:5])
    q = quat.from_matrix(R)
    if scaleFactor!=1:
        scale = [s*scaleFactor for s in scale]
        tr = [t*scaleFactor for t in tr]
    offset=[tr[0],tr[1],tr[2],q[0],q[1],q[2],q[3]]
    return (dim,scale,offset)

def transformToData(scale,offset,timeOffset=0,timeScale=1,isPerspective=False):
    """ Returns a transform, formatted to sofa data given voxelsize, rigid position (offset), time and camera parameters
    """
    return concat(offset[:3])+' '+concat(quat.to_euler(offset[3:])*180./math.pi)+' '+concat(scale)+' '+str(timeOffset)+' '+str(timeScale)+' '+str(int(isPerspective))
