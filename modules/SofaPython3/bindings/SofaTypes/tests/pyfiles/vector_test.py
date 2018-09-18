import math
import unittest

from SofaTypes import Vec1d, Vec2d, Vec3d, Vec4d, Vec4i

class Vec_tests(unittest.TestCase):
    def test_Constructors(self):
        v = Vec1d() # Empty constructor
        self.assertEqual(list(v), [0.0])
        v = Vec1d(math.pi); # argument ctor
        self.assertEqual(list(v), [math.pi])
        v = Vec1d([42.0]); # list ctor
        self.assertEqual(list(v), [42.0])
        p = Vec1d(v) # copy ctor
        self.assertEqual(list(p), [42.0])

        # Test also template specializations
        p = Vec2d(1.0, 2.0)
        self.assertEqual(list(p), [1.0,2.0])
        p = Vec2d([1.0,2.0])
        self.assertEqual(list(p), [1.0,2.0])

        p = Vec3d(1.0, 2.0, 3.0)
        self.assertEqual(list(p), [1.0,2.0,3.0])
        p = Vec3d([1.0,2.0,3.0])
        self.assertEqual(list(p), [1.0,2.0,3.0])

        p = Vec4d(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(list(p), [1.0, 2.0, 3.0, 4.0])
        p = Vec4d([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(list(p), [1.0, 2.0, 3.0, 4.0])

        # too lazy to go to 12.... & also too lazy to do integers

        ## Test also initializers
        v = Vec1d()
        v.set(1.0)
        self.assertEqual(list(v), [1.0])

        v = Vec2d()
        v.set(1.0, 2.0,)
        self.assertEqual(list(v), [1.0, 2.0])

        v = Vec3d()
        v.set(1.0, 2.0, 3.0)
        self.assertEqual(list(v), [1.0, 2.0, 3.0])

        v = Vec4d()
        v.set(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(list(v), [1.0, 2.0, 3.0, 4.0])

        pass

    def test_Acessors(self):
        p = Vec3d(1.0, 2.0, 3.0)
        
        ## [] operators
        p[0] = 5.0
        p[1] = 6.0
        p[2] = 7.0
        self.assertEqual(list(p), [5.0,6.0,7.0])
        ## array limits:
        with self.assertRaises(IndexError):
            p[3]
        
        ## x,y,z accessor.
        self.assertEqual(p.x, 5.0)
        self.assertEqual(p.y, 6.0)
        self.assertEqual(p.z, 7.0)

        ## xy and xyz
        self.assertEqual(p.xy, (5.0,6.0))
        self.assertEqual(p.xyz, (5.0,6.0,7.0))

        ## no Vec4 attributes, please
        self.assertFalse(hasattr(p, "w"))
        self.assertFalse(hasattr(p, "xyzw"))

        p = Vec4i(1, 2, 3, 4)        
        ## [] operators
        p[0] = 5
        p[1] = 6
        p[2] = 7
        p[3] = 8
        self.assertEqual(list(p), [5,6,7,8])
        ## array limits:
        with self.assertRaises(IndexError):
            p[4]
        
        ## x,y,z accessor.
        self.assertEqual(p.x, 5)
        self.assertEqual(p.y, 6)
        self.assertEqual(p.z, 7)
        self.assertEqual(p.w, 8)

        ## xy and xyz
        self.assertEqual(p.xy,   (5,6))
        self.assertEqual(p.xyz,  (5,6,7))
        self.assertEqual(p.xyzw, (5,6,7,8))



    # def test_Operators(self):
    #     v = Vec3d(1,2,3);
        
    # def test_GeometricFunction(self):
    #     ## Norm
    #     p = Vec3d(2.0, 2.0, 2.0)
    #     self.assertAlmostEqual(p.norm(), math.sqrt(12))

    #     ## distanceTo
    #     p1 = Vec3d(2.0, 2.0, 2.0)
    #     p2 = Vec3d(2.0, 2.0, 2.0)
    #     self.assertAlmostEqual(p1.distanceTo(p2), 0.0)

    #     ## distanceTo
    #     p1 = Vec3d(2.0, 2.0, 2.0)
    #     p2 = Vec3d(3.0, 3.0, 3.0)
    #     self.assertAlmostEqual(p1.distanceTo(p2), math.sqrt(3))

    # def test_ScalarMul(self):
    #         p = Vec3d(1.0,2.0,3.0)
    #         p2 = p * 2.0
    #         self.assertTrue(isinstance(p2, Vec3d))
    #         self.assertAlmostEqual((p2-Vec3d(2.0, 4.0, 6.0)).norm(), 0.0)

    # def test_VectorMul(self):
    #         p1 = Vec3d(1.0,2.0,3.0)
    #         p2 = Vec3d(2.0,3.0,4.0)
    #         p3 = p1 * p2
    #         self.assertTrue(isinstance(p3, Vec3d))
    #         self.assertAlmostEqual((p3-Vec3d(1.0*2.0, 2.0*3.0, 3.0*4.0)).norm(), 0.0)

    # def test_VectorDot(self):
    #         p1 = Vec3d(1.0,2.0,3.0)
    #         p2 = Vec3d(2.0,3.0,4.0)
    #         s = p1.dot(p2)
    #         self.assertTrue(isinstance(s, float))
    #         self.assertAlmostEqual(s, 20.0)

    # def test_VectorCross(self):
    #         p1 = Vec3d(1.0,2.0,3.0)
    #         p2 = Vec3d(2.0,3.0,4.0)
    #         p3 = p1.cross(p2)
    #         self.assertTrue(isinstance(p3, Vec3d))
    #         self.assertAlmostEqual((p3-Vec3d(-1.0,2.0,-1.0)).norm(), 0.0)

    # def test_Acessors(self):
    #         p = Vec3d(1.0, 2.0, 3.0)

    #         ## Setters...
    #         p.set(3.0,2.0,1.0)
    #         self.assertEqual(p.toList(), [3.0,2.0,1.0])

    #         ## [] operators
    #         p[0] = 5.0
    #         p[1] = 6.0
    #         p[2] = 7.0
    #         self.assertEqual(p.toList(), [5.0,6.0,7.0])

    #         ## x,y,z accessor.
    #         self.assertEqual(p.x(), 5.0)
    #         self.assertEqual(p.y(), 6.0)
    #         self.assertEqual(p.z(), 7.0)

    #         ## xy and xyz
    #         self.assertEqual(p.xy(), (5.0,6.0))
    #         self.assertEqual(p.xyz(), (5.0,6.0,7.0))



def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Vec_tests)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()

