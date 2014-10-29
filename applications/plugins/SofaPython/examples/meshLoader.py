import Sofa

import SofaPython.Tools

def createScene(node):
    
    obj = node.createChild("obj")
    meshLoader = SofaPython.Tools.meshLoader(obj, "mesh/Armadillo_simplified.obj", scale="10 10 10")
    obj.createObject('VisualModel', name="visual", position="@{0}.position".format(meshLoader.name), triangles="@{0}.triangles".format(meshLoader.name))
    
    vtk = node.createChild("vtk")
    meshLoader = SofaPython.Tools.meshLoader(vtk, "mesh/liver.vtk")
    vtk.createObject('VisualModel', name="visual", position="@{0}.position".format(meshLoader.name), triangles="@{0}.triangles".format(meshLoader.name))
    
    return node
