import Sofa
import SofaImage
import SofaPython.Tools
import sys


def createScene(node):
    node.createObject('RequiredPlugin',name='image')

    global ic
    ic = node.createObject('ImageContainer',template="ImageUC", filename="textures/cubemap_bk.bmp", name="img", transform="-5 -10 0 0 0 0 0.1 0.1 15 0 1 1" , drawBB="1" )
    node.createObject('ImageViewer',template="ImageUC",src="@img")


    # @warning: init is mandatory for projective images to set pinhole camera intrinsic parameters
    ic.init()




    dim = ic.image.getDimensions()
    print "image dimensions",dim



    # directly access to a ImageTransform type
    print ic.transform

    # read access
    print "params",ic.transform.params
    print "translation",ic.transform.translation
    print "rotation",ic.transform.rotation
    print "scale",ic.transform.scale
    print "offsetT",ic.transform.offsetT
    print "scaleT",ic.transform.scaleT
    print "perspective",ic.transform.perspective
    print "camPos",ic.transform.camPos

    # write access i.e. tweaking the image transform
    ic.transform.translation = 0,0,0
    ic.transform.scale = 1,1,100
    print "new translation",ic.transform.translation
    print "new scale",ic.transform.scale



    # image <-> world coordinates conversions
    print 'origin toimage',ic.transform.toImage(0,0,0)


    corner = ic.transform.fromImage(0,0,0)
    node.createObject('MechanicalObject',position=SofaPython.Tools.listToStr(corner), showObject=True, showObjectScale=1, drawMode=1)


    # @warning the image space is defined at the pixel centers (so add a -0.5 pixel offset to get the pixel corners)
    bbox = []
    bbox += ic.transform.fromImage(-0.5,-0.5,-0.5)
    bbox += ic.transform.fromImage(dim[0]-0.5,-0.5,-0.5)
    bbox += ic.transform.fromImage(-0.5,dim[1]-0.5,-0.5)
    bbox += ic.transform.fromImage(dim[0]-0.5,dim[1]-0.5,-0.5)
    bbox += ic.transform.fromImage(-0.5,-0.5,dim[2]-0.5)
    bbox += ic.transform.fromImage(dim[0]-0.5,-0.5,dim[2]-0.5)
    bbox += ic.transform.fromImage(-0.5,dim[1]-0.5,dim[2]-0.5)
    bbox += ic.transform.fromImage(dim[0]-0.5,dim[1]-0.5,dim[2]-0.5)
    node.createObject('MechanicalObject',position=SofaPython.Tools.listToStr(bbox), showObject=True, showObjectScale=10, drawMode=2)


    sys.stdout.flush()

