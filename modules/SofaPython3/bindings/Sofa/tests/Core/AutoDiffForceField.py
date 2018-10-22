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
        ## Position are declared as autodiff numbers
        p = np.array([ adnumber(pos[i], i) for i in range(len(pos)) ])
        u = self.initpos-p
        res = np.array((u * self.ks))
        
        ## This is needed to compute the ad jacobian (in a really ugly way)
        self.res = np.ndarray.flatten(res)
        self.p = np.ndarray.flatten(p)
        
        np.set_printoptions({"all" : lambda x : str(x.x)+"xx, d:"+str(x.d())})
        print("s: ", res)
        
        # To me doing this is fundamentally ugly as we create a matrix full of zero.                 
        self.jacobian = jacobian(self.res, self.p) 
        
        ## Needed to extract the 'number' part of the autodiff array                
        def f(x):
                return x.x
        vf=np.vectorize(f)
               
        out_force += vf(res)
        

    def addDForce(self, df, dx, kFactor, bFactor):
        ## We multiply the big supersparse matrix by the flattened version of x
        tdf = self.jacobian @ (dx.reshape((-1)) * kFactor)
        df += tdf.reshape((-1,3))
        
    def addKToMatrix(self, a, b):
        print(" NOT IMPLEMENTED a,b are non" )

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
                
def createScene(node):
        node.addObject("OglLineAxis")
        node.addObject("RequiredPlugin", name="SofaSparseSolver")
        node.addObject("DefaultAnimationLoop", name="loop")
        node.addObject("EulerImplicit")
        node.addObject("CGLinearSolver", tolerance=1e-12, threshold=1e-12)
       
        a=node.addChild( RestShapeObject( MyForceField("customFF", ks=5.0) , name="python", position=[[i-7.5, 0, 0] for i in range(10)] ) )
        a.mechanical.showColor = [1.0,0.0,0.0,1.0]
        
        b=node.addChild( RestShapeObject( CreateObject("RestShapeSpringsForceField", stiffness=5.0) , name="c++", position=[[i-2.5, 0, 0] for i in range(10)]))
        b.mechanical.showColor = [1.0,1.0,0.0,1.0]

######################################### TESTS ####################################################
## In the following is the code used to consider this example as a test.
####################################################################################################
class Test(unittest.TestCase):
    def test_example(self):
            createScene(Sofa.Node("root"))

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
