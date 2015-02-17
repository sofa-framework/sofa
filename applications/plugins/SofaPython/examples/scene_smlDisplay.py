import Sofa

import SofaPython.sml

def createScene(rootNode):
    model = SofaPython.sml.Model("smlSimple.xml")
    scene = SofaPython.sml.SceneDisplay(rootNode,model)
    scene.param.colorByTag["tag02"]="1. 1. 0."
    scene.createScene()
    
