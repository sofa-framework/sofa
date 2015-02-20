import os.path

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
        imageObject = parentNode.createObject('ImageContainer', template=imageTemplate, name=name, cacheFile, drawBB=False)
    else:
        imageObject = insertImageFunc(**args)
        assert imageObject.name == name, "image.Tools.insertImageCache: image name mismatch"
        parentNode.createObject('ImageExporter', template=imageTemplate, image="@"+image, transform="@"+transform, filename=cacheFile, exportAtEnd="1", printLog="1")
    return imageObject
