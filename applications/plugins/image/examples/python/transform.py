import Sofa
import SofaImage
import SofaPython.Tools
import sys


def createScene(node):
    node.createObject('RequiredPlugin',name='image')

    global ic
    ic = node.createObject('ImageContainer',template="ImageUC", filename="textures/cubemap_bk.bmp", name="img", transform="-5 -10 0 0 0 0 0.1 0.1 15 0 1 1" , drawBB="1" )
    node.createObject('ImageViewer',template="ImageUC",src="@img")


    print ic.transform

    corner = ic.transform.fromImage(0,0,0)

    print ic.transform.toImage(0,0,0)


    node.createObject('MechanicalObject',position=SofaPython.Tools.listToStr(corner), showObject=True, showObjectScale=1, drawMode=1)


    sys.stdout.flush()

