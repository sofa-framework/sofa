import Sofa
import SofaPython.sceneStats

def createScene(node):

    child0 = node.createChild("child0")
    child00 = child0.createChild("child00")
    child01 = child0.createChild("child01")
    child000 = child00.createChild("child000")
    child01.addChild(child000)

    child0.createObject("MechanicalObject")



    SofaPython.sceneStats.printSceneStats(node)