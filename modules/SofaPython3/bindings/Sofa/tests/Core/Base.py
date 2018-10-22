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

def runTests():
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()

def createScene(rootNode):
        runTests()
