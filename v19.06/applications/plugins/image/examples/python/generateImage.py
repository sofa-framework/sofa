import Sofa
import SofaImage
import sys


def createScene(node):
    node.createObject('RequiredPlugin',name='image')

    dimx,dimy,dimz,dimc,dimt = 10, 10, 1, 3, 1

    ic = node.createObject('GenerateImage',template="ImageD", name="img", dim=str(dimx)+" "+str(dimy)+" "+str(dimz)+" "+str(dimc)+" "+str(dimt) )
    ic.init()
    img_numpy = SofaImage.image_as_numpy( ic.image, 0)
    for z in xrange(dimz):
        for y in xrange(dimy):
            for x in xrange(dimx):
                img_numpy[:,z,y,x]=[x+y,x-y,-x+y]
    print img_numpy
    node.createObject('ImageViewer',template="ImageD", src="@img", transform="0 0 0 0 0 0 1 1 1 0 1 0")

    ic2 = node.createObject('GenerateImage',template="ImageUC", name="img2", dim=str(dimx)+" "+str(dimy)+" "+str(dimz)+" "+str(dimc)+" "+str(dimt) )
    ic2.init()
    img_numpy = SofaImage.image_as_numpy( ic2.image, 0)
    for z in xrange(dimz):
        for y in xrange(dimy):
            for x in xrange(dimx):
                for c in xrange(dimc):
                    img_numpy[c,z,y,x]=x+y
    print img_numpy
    node.createObject('ImageViewer',template="ImageUC", src="@img2", transform=str(dimx+2)+" 0 0 0 0 0 1 1 1 0 1 0")

    sys.stdout.flush()

    return 0
