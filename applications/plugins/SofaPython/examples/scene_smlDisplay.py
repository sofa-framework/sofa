import Sofa

import SofaPython.sml

def createScene(rootNode):
    model = SofaPython.sml.Model("smlSimple.xml")
    scene = SofaPython.sml.SceneDisplay(rootNode,model)
    scene.createScene()
    
