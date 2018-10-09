import unittest
import Sofa
import numpy
import ad
from ad import *
from Sofa.Simulation import SingleSimulation

class MyForceField(Sofa.Core.BaseForceField):
    def __init__(self, *args, **kwargs):
        kwargs["ks"] = kwargs.get("ks", 0.1)
        kwargs["kd"] = kwargs.get("kd", 1.0)
        Sofa.Core.BaseForceField.__init__(self, *args, **kwargs)
                        
    def init(self):
        self.initpos = self.mstate.position.toarray().copy()
        
    def addForce(self, m, out_force, pos, vel):
        """This is not a super clever mechanical law but...well it does the job"""
        ip = adnumber(self.initpos)
        ks = adnumber(self.ks)
        kd = adnumber(self.kd)
        of = adnumber(out_force)
        p = adnumber(pos)
        v = adnumber(vel)
        u = ip-p
        res = of + (u * ks - v * kd)
        print("RES: "+str(u.p))
        #u = self.initpos - pos
        #out_force = op + (u * self.ks - vel * self.kd)
        print(" Python::addForce: ")

    def addDForce(self, v, dx):
        print(" Python::addDForce: ", a, " ", b)

    def addMBKdx(self, a, b):
        print(" Python::addMBKdx: ", a, " ", b)

    #def updateForceMask(self):
    #    print(" Python::updateFroceMask: ")

class Test(unittest.TestCase):
    def test_animation(self):
        node = Sofa.Node("TestAnimation")
        node.addObject("OglLineAxis")
        node.addObject("RequiredPlugin", name="SofaSparseSolver")
        node.addObject("DefaultAnimationLoop", name="loop")
        node.addObject("EulerImplicit")
        node.addObject("CGLinearSolver")
        object1 = node.addChild("object1")
        c = object1.addObject("MechanicalObject", position=[[i-5.0, 0, 0] for i in range(10)])
        m = object1.addObject("UniformMass", vertexMass=0.1)
        d = object1.addObject(MyForceField("customFF", ks=10.0))
        
        c.showObject = True
        c.drawMode = 1

        #SingleSimulation.init(node)
        #for i in range(10):
        #    SingleSimulation.animate(node, 0.01)

        return node

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
        rootNode.addChild(Test().test_animation())
