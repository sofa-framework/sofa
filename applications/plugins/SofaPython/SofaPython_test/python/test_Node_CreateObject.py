import Sofa

import SofaTest

def createScene(rootNode):
    
    # simple input string
    node = rootNode.createChild("nodeA")
    node.createObject('MechanicalObject', template="Vec3", name="dof", position="1 1 1 2 2 2 3 3 3")
    node.createObject('DilateEngine', template="Vec3", name="dilate", thickness="1.2 2.3 3.4")
    node.createObject('PythonScriptController',
        filename = __file__,
        classname = 'TestDataSerialization',
        name = 'script' )
    
    # simple input string, space before
    node = rootNode.createChild("nodeA")
    node.createObject('MechanicalObject', template="Vec3", name="dof", position="1 1 1 2 2 2 3 3 3 ")
    node.createObject('DilateEngine', template="Vec3", name="dilate", thickness="1.2 2.3 3.4 ")
    node.createObject('PythonScriptController',
        filename = __file__,
        classname = 'TestDataSerialization',
        name = 'script' )

    # simple input string, space at the end
    node = rootNode.createChild("nodeA")
    node.createObject('MechanicalObject', template="Vec3", name="dof", position="1 1 1 2 2 2 3 3 3 ")
    node.createObject('DilateEngine', template="Vec3", name="dilate", thickness="1.2 2.3 3.4 ")
    node.createObject('PythonScriptController',
        filename = __file__,
        classname = 'TestDataSerialization',
        name = 'script' )

    position = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    thickness = [1.2, 2.3, 3.4]
    # input string with delimiters
    node = rootNode.createChild("nodeB")
    node.createObject('MechanicalObject', template="Vec3", name="dof", position=repr(position))
    node.createObject('DilateEngine', template="Vec3", name="dilate", thickness=repr(thickness))
    node.createObject('PythonScriptController',
        filename = __file__,
        classname = 'TestDataSerialization',
        name = 'script' )
    
    # input python list
    node = rootNode.createChild("nodeB")
    node.createObject('MechanicalObject', template="Vec3", name="dof", position=position)
    node.createObject('DilateEngine', template="Vec3", name="dilate", thickness=thickness)
    node.createObject('PythonScriptController',
        filename = __file__,
        classname = 'TestDataSerialization',
        name = 'script' )

    

class TestDataSerialization(SofaTest.Controller):

    def onLoaded(self, node):
        SofaTest.Controller.onLoaded(self, node)
        self.dof = self.node.getObject("dof")
        self.dilate = self.node.getObject("dilate")

    def onEndAnimationStep(self, dt):
        self.ASSERT("[[1, 1, 1], [2, 2, 2], [3, 3, 3]]" == self.dof.findData('position').getValueString(), "test_vec_Vec3")
        self.ASSERT("[1.2, 2.3, 3.4]" == self.dilate.findData('thickness').getValueString(), "test_vec_Real")
        self.sendSuccess()
