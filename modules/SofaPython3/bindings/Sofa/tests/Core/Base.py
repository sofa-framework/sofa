# coding: utf8

import Sofa
import unittest

class Test(unittest.TestCase):
    def test_data_property(self):
            root = Sofa.Node("rootNode")
            c = root.createObject("MechanicalObject", name="t", position=[[0,0,0],[1,1,1],[2,2,2]])
            self.assertTrue(hasattr(c, "__data__"))
            self.assertGreater(len(c.__data__), 0)
            self.assertTrue("name" in c.__data__)
            self.assertTrue("position" in c.__data__)
            self.assertFalse(hasattr(c.__data__, "invalidEntry"))
            self.assertTrue( isinstance(c.__data__, Sofa.Core.DataDict))

    def test_getsetData(self):
            root = Sofa.Node("root")
            c = root.createObject("MechanicalObject", name="t", position=[[0,0,0],[1,1,1],[2,2,2]])
            c.setToData("newdata", 1.0)
            c.setToDict("newdict", 1.0)

            self.assertTrue(isinstance(c.newdata, float))
            self.assertTrue(isinstance(c.newdict, float))

            self.assertTrue(isinstance(c.getFromData("newdata"), float))
            self.assertTrue(isinstance(c.getFromDict("newdict"), float))


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
