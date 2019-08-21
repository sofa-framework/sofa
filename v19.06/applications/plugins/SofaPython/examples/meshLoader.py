import Sofa

import SofaPython.Tools

def createScene(node):
    
    obj = node.createChild("obj")
    meshLoader = SofaPython.Tools.meshLoader(obj, "mesh/Armadillo_simplified.obj", scale="10 10 10")
    obj.createObject('VisualModel', name="visual", src="@"+meshLoader.name)
    
    vtk = node.createChild("vtk")
    meshLoader = SofaPython.Tools.meshLoader(vtk, "mesh/liver.vtk")
    vm = vtk.createObject('VisualModel', name="visual", src="@"+meshLoader.name)
    vm.setColor(1,0,0,.5)
    
    return node
