# coding: utf8
import unittest
import Sofa
from Sofa.Types import Vec3

class Test(unittest.TestCase):
    def test_constructor(self):
        v0 = Sofa.Types.Vec3()
        v1 = Sofa.Types.Vec3([1.0,1.0,1.0])
        v2 = v0 + v1
        self.assertTrue( isinstance(v2, Sofa.Types.Vec3 ) )
        self.assertEqual( v2, v1 )

    def test_wrapAround(self):
        n = Sofa.Node("node")
        m = n.addObject("MechanicalObject", position=[[1.0,1.1,1.2],[2.0,2.1,2.2],[3.0,3.1,3.2]])
        c = Vec3(m.position)
        print("Vec3, ", c)

def runTests():
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
