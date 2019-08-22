import SofaPython.Tools

def createSceneAndController(rootNode):
    #rootNode.createObject('VisualStyle', displayFlags='showBehaviorModels showVisual')
    dof = rootNode.createObject('MechanicalObject', template='Vec3d', name='dofs', size=531, showObject=True, drawMode=1)
    global dofDataIO
    dofDataIO = SofaPython.Tools.ComponentDataIO(dof, ["position"])
    
def bwdInitGraph(node):
    dofDataIO.readData()
        