import Sofa

import SofaTest

def createScene(rootNode):

    externalComponent = rootNode.createObject( 'ExternalComponent', name="pouet" )
    externalComponent.helloWorld()

    c = rootNode.createObject('PythonScriptController',
        filename = __file__,
        classname = 'TestDataSerialization',
        name = 'script' )
    print "DEBUG", c


class TestDataSerialization(SofaTest.Controller):

    def onLoaded(self, node):
        SofaTest.Controller.onLoaded(self, node)
        
        # simple input string
        nodeA = node.createChild("nodeA")
        moA = nodeA.createObject('MechanicalObject', template="Vec3", position="1 1 1 2 2 2 3 3 3")
        print "DEBUG", moA.findData('position').getValueString()
        self.should("[ 1 1 1, 2 2 2, 3 3 3 ]" == moA.findData('position').getValueString())
        
        # input string with delimiters
        nodeB = node.createChild("nodeB")
        moB = nodeB.createObject('MechanicalObject', template="Vec3", position="[1 1 1, 2 2 2, 3 3 3]")
        print "DEBUG", moB.findData('position').getValueString()
        self.should("[ 1 1 1, 2 2 2, 3 3 3 ]" == moB.findData('position').getValueString())
