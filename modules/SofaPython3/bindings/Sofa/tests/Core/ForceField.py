import unittest
import Sofa
import numpy
from Sofa.Simulation import SingleSimulation

class MyForceField(Sofa.Core.BaseForceField):
    def __init__(self, *args, **kwargs):
        Sofa.Core.BaseForceField.__init__(self, *args, **kwargs)

    def init(self):
        print(" Python::init() at ", self.name)

    def addForce(self, m, f, x, v):
        print(" Python::addForce: ", m, " ", f, " ", x ," ", v)

        n = numpy.ones((1000,1))
        with f.writeable() as fw:
            fw += n

        print(" Python::addForce: ", m, " ", f, " ", x ," ", v)


    def addDForce(self, a, b):
        print(" Python::addForce: ", a, " ", b)

    def addMBKdx(self, a, b):
        print(" Python::addMBKdx: ", a, " ", b)

    def updateForceMask(self):
        print(" Python::updateFroceMask: ")

class Test(unittest.TestCase):
    def test_animation(self):
        node = Sofa.Node("root")
        node.addObject("DefaultAnimationLoop", name="loop")
        node.addObject("EulerImplicit")
        node.addObject("CGLinearSolver")
        object1 = node.addChild("object1")
        c = object1.addObject("MechanicalObject", position=[0,0,0]*1)
        d = object1.addObject(MyForceField("customFF"))

        SingleSimulation.init(node)
        for i in range(10):
            SingleSimulation.animate(node, 0.01)

def getTestsName():
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    return [ test.id().split(".")[2] for test in suite]

def runTests():
        import sys
        suite = None
        if( len(sys.argv) == 1 ):
            suite = unittest.TestLoader().loadTestsFromTestCase(Test)
        else:
            suite = unittest.TestSuite()
            suite.addTest(Test(sys.argv[1]))
        return unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()

def createScene(rootNode):
        runTests()

