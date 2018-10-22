# coding: utf8

import unittest
import Sofa
import numpy as np
import ad
from ad import *
from Sofa.Simulation import SingleSimulation

class MyForceField(Sofa.ForceField):
    def __init__(self, *args, **kwargs):
        kwargs["ks"] = kwargs.get("ks", 1.0)
        kwargs["kd"] = kwargs.get("kd", 0.1)
        Sofa.ForceField.__init__(self, *args, **kwargs)
                        
    def init(self):
        self.initpos = self.mstate.position.toarray().copy()
        self.k = np.zeros((1,1))
        self.f = []
        self.d = 0.5

    def addForce(self, m, out_force, pos, vel):
        ip = self.initpos
        ks = self.ks 
        kd = self.kd 
        of = out_force
        p = pos #adnumber(pos, "pos")
        v = vel
        u = ip-p
        res = (u * ks ) 
        
        #self.res = np.ndarray.flatten(res)
        #self.p = np.ndarray.flatten(p)
                        
        #def f(x):
        #        return x.x
        #vf=np.vectorize(f)
        
        #out_force += vf(res)
        out_force += res
        
        #print(" Python::addForce: ", u, "*", ks, "=", out_force)
         

    def addDForce(self, df, dx, kFactor, bFactor):
        #print("===============================")
        #print(" F", self.res)
        #print(" pos", self.p)
        #print(" Python::addDForce df(in): ", df)
        #print(" Python::addDForce dx: ", np.ndarray.flatten(dx))        
        #print(" Python::addDForce kFactor: ", (kFactor, bFactor))        
        #print("RES is ", self.res[0,0].d())
        #print("RES is ", self.res[0,1].d())
        #print("RES is ", self.res[0,2].d())
        return 
        from ad import jacobian
        j = jacobian(self.res, self.p)
        #print(" Python::addDForce J:", j)
        
        tdf = j @ (np.ndarray.flatten(dx) * kFactor)
        #print(" Python::addDForce df: ", tdf)
        df += tdf.reshape((-1,3))
        print(" Python::addDForce df: ", df)
        
    def addKToMatrix(self, a, b):
        print(" Python::addKToMatrix: ", a, " ", b)

    #def updateForceMask(self):
    #    print(" Python::updateFroceMask: ")

class CreateObject(object):
        def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

def RestShapeObject(impl, name="unnamed", position=[]):
        node = Sofa.Node(name)
        c = node.addObject("MechanicalObject", name="mechanical", position=position)
        c.showObject = True
        c.drawMode = 1

        m = node.addObject("UniformMass", name="mass", vertexMass=0.1)
        
        if isinstance(impl, CreateObject): 
                node.createObject(*impl.args, **impl.kwargs)        
        else:        
                d = node.addObject(impl)
        return node
                
class Test(unittest.TestCase):
    def test_animation(self):
        node = Sofa.Node("TestAnimation")
        node.addObject("OglLineAxis")
        node.addObject("RequiredPlugin", name="SofaSparseSolver")
        node.addObject("DefaultAnimationLoop", name="loop")
        node.addObject("EulerImplicit")
        node.addObject("CGLinearSolver", tolerance=1e-12, threshold=1e-12)
        #node.addObject("SparseLDLSolver")

        #object1.addChild( MyForceField("customFF", ks=5.0) )
        a=node.addChild( RestShapeObject( MyForceField("customFF", ks=5.0) , name="python", position=[[i*1-10.0, 0, 0] for i in range(200)] ) )
        a.mechanical.showColor = [1.0,0.0,0.0,1.0]
        b=node.addChild( RestShapeObject( CreateObject("RestShapeSpringsForceField", stiffness=5.0) , name="c++", position=[[i*0.5-1.0, 0, 0] for i in range(1)]))
        b.mechanical.showColor = [1.0,1.0,0.0,1.0]
        
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
