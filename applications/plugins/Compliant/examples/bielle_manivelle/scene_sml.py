import os.path

import Sofa

import Compliant.sml

def createScene(node):
    scene_bielle_manivelle = Compliant.sml.Scene(os.path.join(os.path.dirname(__file__),"bielle_manivelle.xml"))
    scene_bielle_manivelle.param.showRigid=True
    scene_bielle_manivelle.param.showOffset=True
    scene_bielle_manivelle.createScene(node)
    return node

