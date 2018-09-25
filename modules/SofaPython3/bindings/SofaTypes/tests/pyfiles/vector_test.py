import math
import unittest

import SofaTypes
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

    def test_Accessors(self):
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

    def test_GeometricFunction(self):
        ## Norm
        p = Vec3d(2.0, 2.0, 2.0)

        self.assertEqual(6.0, sum(p)) # neat!
            
        p.clear()
        self.assertEqual(list(p), [0.0, 0.0, 0.0])
        p.fill(2.0)
        self.assertEqual(list(p), [2.0, 2.0, 2.0])
        
        p = Vec3d(2.0, 2.0, 2.0)
        self.assertAlmostEqual(p.norm(), math.sqrt(12))
        self.assertAlmostEqual(p.norm2(), 12.0)
        self.assertAlmostEqual(p.lNorm(1), 6.0)
        p1 = p.normalized()
        self.assertEqual(p1, Vec3d(0.57735, 0.57735, 0.57735))
        ret = p.normalize()
        self.assertEqual(ret, True)
        self.assertEqual(p, Vec3d(0.57735, 0.57735, 0.57735))
        
        ## dot product
        p1 = Vec3d(1.0, 3.0, -5.0)
        p2 = Vec3d(4.0, -2.0, -1.0)
        self.assertAlmostEqual(p1.dot(p2), 3.0)
        self.assertEqual(list(p1.cross(p2)), [-13.0, -19.0, -14.0])
        
        self.assertTrue(hasattr(Vec2d(), "cross"))
        self.assertTrue(hasattr(Vec3d(), "cross"))
        self.assertFalse(hasattr(Vec4d(), "cross"))
        
    def test_Operators(self):
        v1 = Vec3d(1,2,3);
        v2 = Vec3d(2,3,4);

        self.assertTrue(v1 != v2)
        self.assertTrue(v1 == v1) 

        self.assertAlmostEqual(v1 * v2, 20.0) # dot product
        self.assertEqual(list(v1 + v2), [3.0, 5.0, 7.0]) # vector addition
        self.assertEqual(list(v1 - v2), [-1.0, -1.0, -1.0]) # vector substraction
        
        self.assertEqual(list(v1 * 2.0), [2.0, 4.0, 6.0])
        self.assertEqual(list(v1 * 2), [2.0, 4.0, 6.0])
        tmp = Vec3d(list(v1))
        tmp *= 2.0  ## Because :)
        self.assertEqual(list(tmp),[2.0, 4.0, 6.0])
        tmp = Vec3d(list(v1))
        tmp *= 2 ## Because
        self.assertEqual(list(tmp),[2.0, 4.0, 6.0])
        
        self.assertEqual(list(v1 / 2.0), [0.5, 1.0, 1.5])
        self.assertEqual(list(v1 / 2), [0.5, 1.0, 1.5])
        tmp = Vec3d(list(v1))
        tmp /= 2.0
        self.assertEqual(list(tmp),[0.5, 1.0, 1.5])
        tmp = Vec3d(list(v1))
        tmp /= 2
        self.assertEqual(list(tmp),[0.5, 1.0, 1.5])
        
        
def createScene(rootNode):
    suite = unittest.TestLoader().loadTestsFromTestCase(Vec_tests)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()

