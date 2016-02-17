import Sofa

def createScene(rootNode):
    rootNode.createObject('PythonScriptController', name="controller", filename=__file__, classname="TestController")

class TestController(Sofa.PythonScriptController):

    def getTrue(self):
        return True

    def getFalse(self):
        return False

    def getInt(self):
        return 7

    def getFloat(self):
        return 12.34

    def getString(self):
        return "test string"

    def getNone(self):
        return None

    def getNothing(self):
        pass

    def add(self,a,b):
        print "add ", a, b, a+b
        return a+b

