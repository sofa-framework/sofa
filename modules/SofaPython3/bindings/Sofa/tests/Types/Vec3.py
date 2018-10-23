# coding: utf8

import Sofa
import unittest

class Test(unittest.TestCase):
    def test_constructor(self):
        v0 = Sofa.Types.Vec3()
        v1 = Sofa.Types.Vec3([1.0,1.0,1.0])
        v2 = v0 + v1
        self.assertTrue( isinstance(v2, Sofa.Types.Vec3 ) )
        self.assertEqual( v2, v1 )

def runTests():
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
