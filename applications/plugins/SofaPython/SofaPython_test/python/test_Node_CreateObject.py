import Sofa

import SofaTest

def createScene(rootNode):
    
    # simple input string
    node = rootNode.createChild("nodeA")
    node.createObject('MechanicalObject', template="Vec3", name="dof", position="1 1 1 2 2 2 3 3 3")
    node.createObject('PythonScriptController',
        filename = __file__,
        classname = 'TestDataSerialization',
        name = 'script' )
    
    position = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    # input string with delimiters
    node = rootNode.createChild("nodeB")
    node.createObject('MechanicalObject', template="Vec3", name="dof", position=repr(position))
    node.createObject('PythonScriptController',
        filename = __file__,
        classname = 'TestDataSerialization',
        name = 'script' )
    
    # input python list
    node = rootNode.createChild("nodeB")
    node.createObject('MechanicalObject', template="Vec3", name="dof", position=position)
    node.createObject('PythonScriptController',
        filename = __file__,
        classname = 'TestDataSerialization',
        name = 'script' )

    

class TestDataSerialization(SofaTest.Controller):

    def onLoaded(self, node):
        SofaTest.Controller.onLoaded(self, node)
        self.dof = self.node.getObject("dof")

    def onEndAnimationStep(self, dt):
        self.ASSERT("[[1, 1, 1], [2, 2, 2], [3, 3, 3]]" == self.dof.findData('position').getValueString(), "test1")
        self.sendSuccess()
