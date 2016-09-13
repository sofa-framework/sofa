import Sofa
import SofaImage
import sys


def createSceneAndController(node):
    node.createObject('RequiredPlugin',name='image')

    global ic
    ic = node.createObject('ImageContainer',template="ImageUC", filename="textures/lights2.png", name="img")
    print ic.image

    img_numpy = SofaImage.image_as_numpy( ic.image )
    print img_numpy


    ## t,[r,g,b],z,y,x

    print img_numpy[0][:,0,0,0]
    img_numpy[0][:,0,0,0] = [100,101,102]
    print img_numpy[0][:,0,0,0]

    sys.stdout.flush()

    return 0


def bwdInitGraph(root):

    # set intensities in a rectangle
    img_numpy0 = SofaImage.image_as_numpy( ic.image,0 )
    dims,dimz,dimy,dimx = img_numpy0.shape
    for z in xrange(dimz):
        for y in range(800,dimy):
            for x in xrange(200,dimx):
                img_numpy0[:,z,y,x]=[x%255,0,0]

    n = root.createChild("visu")
    n.createObject('ImageViewer', template="ImageUC", image="@../img.image", transform="@../img.transform")
    n.init()