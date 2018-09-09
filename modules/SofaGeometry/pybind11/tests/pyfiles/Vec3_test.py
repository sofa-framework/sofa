import math
import unittest
from SofaGeometry import Vec3

class Vec3_tests(unittest.TestCase):
    def test_Constructors(self):
        ## Empty constructors
        p = Vec3()
        self.assertEqual(p.toList(), [0.0,0.0,0.0])

        ## From float or int.
        p = Vec3(1.0, 2.0, 3.0)
        self.assertEqual(p.toList(), [1.0,2.0,3.0])

        ## Copy from list
        p = Vec3([1.0,2.0,3.0])
        self.assertEqual(p.toList(), [1.0,2.0,3.0])

        ## Copy from other vector
        p2 = Vec3(p)
        self.assertEqual(p.toList(), p2.toList())

    def test_GeometricFunction(self):
        ## Norm
        p = Vec3(2.0, 2.0, 2.0)
        self.assertAlmostEqual(p.norm(), math.sqrt(12))

        ## distanceTo
        p1 = Vec3(2.0, 2.0, 2.0)
        p2 = Vec3(2.0, 2.0, 2.0)
        self.assertAlmostEqual(p1.distanceTo(p2), 0.0)

        ## distanceTo
        p1 = Vec3(2.0, 2.0, 2.0)
        p2 = Vec3(3.0, 3.0, 3.0)
        self.assertAlmostEqual(p1.distanceTo(p2), math.sqrt(3))

    def test_ScalarMul(self):
        p = Vec3(1.0,2.0,3.0)
        p2 = p * 2.0
        self.assertTrue(isinstance(p2, Vec3))
        self.assertAlmostEqual((p2-Vec3(2.0, 4.0, 6.0)).norm(), 0.0)

    def test_VectorMul(self):
        p1 = Vec3(1.0,2.0,3.0)
        p2 = Vec3(2.0,3.0,4.0)
        p3 = p1.linearmul(p2)
        self.assertTrue(isinstance(p3, Vec3))
        self.assertAlmostEqual((p3-Vec3(1.0*2.0, 2.0*3.0, 3.0*4.0)).norm(), 0.0)

    def test_VectorDot(self):
        p1 = Vec3(1.0,2.0,3.0)
        p2 = Vec3(2.0,3.0,4.0)
        s = p1.dot(p2)
        s2 = p1 * p2
        self.assertTrue(isinstance(s, float))
        self.assertAlmostEqual(s, 20.0)
        self.assertAlmostEqual(s, s2)

    def test_VectorCross(self):
        p1 = Vec3(1.0,2.0,3.0)
        p2 = Vec3(2.0,3.0,4.0)
        p3 = p1.cross(p2)
        self.assertTrue(isinstance(p3, Vec3))
        self.assertAlmostEqual((p3-Vec3(-1.0,2.0,-1.0)).norm(), 0.0)
        
    def test_Acessors(self):
        p = Vec3(1.0, 2.0, 3.0)
        
        ## Setters...
        p.set(3.0,2.0,1.0)
        self.assertEqual(p.toList(), [3.0,2.0,1.0])
        
        ## [] operators
        p[0] = 5.0
        p[1] = 6.0
        p[2] = 7.0
        self.assertEqual(p.toList(), [5.0,6.0,7.0])
        
        ## x,y,z accessor.
        self.assertEqual(p.x, 5.0)
        self.assertEqual(p.y, 6.0)
        self.assertEqual(p.z, 7.0)
        
        ## xy and xyz
        self.assertEqual(p.xy, (5.0,6.0))
        self.assertEqual(p.xyz, (5.0,6.0,7.0))
        
        

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Vec3_tests)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
