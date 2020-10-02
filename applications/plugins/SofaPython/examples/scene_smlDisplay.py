import Sofa

import SofaPython.sml

def createScene(rootNode):
    model = SofaPython.sml.Model("smlSimple.sml")
    scene = SofaPython.sml.SceneDisplay(rootNode,model)
    scene.param.colorByTag["default"]="0.8 0.8 0.8"
    scene.param.colorByTag["red"]="1. 0. 0."
    scene.param.colorByTag["green"]="0. 1. 0."
    scene.createScene()
    
