import SofaPython.Tools

def createSceneAndController(rootNode):
    rootNode.createObject('MeshObjLoader', name='loader', filename='mesh/Armadillo_verysimplified.obj')
    dof = rootNode.createObject('MechanicalObject', name='dofs', src='@loader', showObject=True, drawMode=1)
    global dofDataIO
    dofDataIO = SofaPython.Tools.ComponentDataIO(dof, ["position"])
    
def bwdInitGraph(node):
    dofDataIO.writeData()
    