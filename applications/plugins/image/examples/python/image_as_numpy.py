import Sofa
import SofaImage
import sys


def createScene(node):
    node.createObject('RequiredPlugin',name='image')

    ic = node.createObject('ImageContainer',template="ImageUC", filename="textures/lights2.png")
    print ic.image

    img_numpy = SofaImage.image_as_numpy( ic.image )
    print img_numpy


    ## t,x,y,z,[r,g,b]

    print img_numpy[0][0][0][0]
    img_numpy[0][0][0][0] = [100,101,102]
    print img_numpy[0][0][0][0]

    sys.stdout.flush()

    return 0