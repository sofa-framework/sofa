# coding: utf8

import unittest
import Sofa

class Test(unittest.TestCase):
        def __init__(self,a):
                unittest.TestCase.__init__(self,a)
                    
        def test_createObjectWithParam(self):
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=[[0,0,0],[1,1,1],[2,2,2]])
                self.assertTrue( c is not None )
        
        def test_createObjectWithInvalidParamName(self):
                ## This one should raise an error because of 'v' should rise a type error.
                root = Sofa.Node("rootNode")
                self.assertRaises(TypeError, root.createObject, "MechanicalObject", name="tt", v=[[0,0,0],[1,1,1],[2,2,2]])

        def test_createObjectWithInvalidParamValue(self):
                ## This one should raise an error because of 'v' should rise a type error.
                root = Sofa.Node("rootNode")
                root.createObject("MechanicalObject", name="tt", position="xmoi")
                self.fail("We should find a solution not to emit a warning but an exception")

        def test_data_property(self):
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=[[0,0,0],[1,1,1],[2,2,2]])
                self.assertTrue(hasattr(c, "__data__"))
                self.assertGreater(len(c.__data__), 0)
                self.assertTrue("name" in c.__data__)
                self.assertTrue("position" in c.__data__)
                self.assertFalse(hasattr(c.__data__, "invalidEntry"))
                self.assertTrue(isinstance(c.__data__, Sofa.Core.DataDict))

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
