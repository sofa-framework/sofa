import unittest
from vec3 import *

class Vec3_test(unittest.TestCase):

    def test_equal(self):
        v1 = Vec3(0.)
        v2 = Vec3(-1.)
        self.assertEqual(v1,v1)
        self.assertNotEqual(v1,v2)

        self.assertEqual(v1,[0.,0.,0.])
        self.assertNotEqual(v1,[-1.,-1.,-1.])
        self.assertEqual(v2,[-1.,-1.,-1.])

        v = Vec3(0.,-1.,0.)
        self.assertEqual(v,[0.,-1.,0.])


    def test_constructors(self):
        v = Vec3()
        self.assertTrue(len(v),3)
        self.assertEqual(v, [0.,0.,0.])

        v = Vec3(1.)
        self.assertEqual(v, [1.,1.,1.])

        v = Vec3(1.,2.,3.)
        self.assertEqual(v, [1.,2.,3.])

        v = Vec3([1,2,3])
        self.assertEqual(v, [1.,2.,3.])

        v1 = Vec3(2,2,2)
        v = Vec3(v1)
        self.assertEqual(v, v1)

        # If args are not expected should print the doc
        # v = Vec3(1,2)


## PUBLIC METHODS


    def test_getNorm(self):
        v = Vec3(1.,2.,2.)
        self.assertEqual(v.getNorm(), 3.)

    def test_toString(self):
        v = Vec3(1.,2.,2.)
        self.assertEqual(v.toString(), "1.0 2.0 2.0")

    def test_normalize(self):
        v = Vec3(1.,2.,2.)
        v.normalize()
        self.assertEqual(v, [1./3.,2./3.,2./3.])

    def test_translate(self):
        v = Vec3()
        v.translate(1.)
        self.assertEqual(v, [1.,1.,1.])

        v.translate(1.,2.,3.)
        self.assertEqual(v, [2.,3.,4.])

        v.translate([1.,2.,3.])
        self.assertEqual(v, [3.,5.,7.])

        # Should also work with operators '+' and '-'
        v1 = Vec3(1.)
        v2 = Vec3(1.)
        scalar = 1.
        self.assertEqual(v1+v2,[2.,2.,2.])
        self.assertEqual(v1+scalar,[2.,2.,2.])
        self.assertEqual(v1-v2,[0.,0.,0.])
        self.assertEqual(v1-scalar,[0.,0.,0.])

        # If args are not expected should print the doc
        # v.translate(1,2)

    def test_scale(self):
        v = Vec3(1.,1.,1.)
        v.scale(2.)
        self.assertEqual(v, [2.,2.,2.])

        v.scale(1.,2.,3.)
        self.assertEqual(v, [2.,4.,6.])

        v.scale([1.,2.,3.])
        self.assertEqual(v, [2.,8.,18.])

        # Should also work with operators '*' and '/'
        v1 = Vec3(1)
        v2 = Vec3(2.)
        scalar = 2.
        self.assertEqual(v1*v2,[2.,2.,2.])
        self.assertEqual(v1*scalar,[2.,2.,2.])
        self.assertEqual(v1/v2,[0.5,0.5,0.5])
        self.assertEqual(v1/scalar,[0.5,0.5,0.5])

        # If args are not expected should print the doc
        # v.scale(2,1)

    def test_rotateFromQuat(self):
        from quat import Quat
        from math import pi

        v = Vec3(1.,1.,1.)
        q = Quat.createFromAxisAngle(Vec3([1.,0.,0.]),pi/2.)
        v.rotateFromQuat(q)

        self.assertEqual(v[0],1.)
        self.assertEqual(math.floor(v[1]),-1.)
        self.assertEqual(v[2],1.)

    def test_rotateFromEuler(self):
        from math import pi

        v = Vec3(1.,1.,1.)
        v.rotateFromEuler([pi/2.,0.,0.])

        self.assertAlmostEqual(v[0],1.)
        self.assertAlmostEqual(v[1],-1.)
        self.assertAlmostEqual(v[2],1.)

        v = Vec3(1.,1.,1.)
        v.rotateFromEuler([pi/2.,-pi/2.,0.],"rxyz")

        self.assertAlmostEqual(v[0],-1.)
        self.assertAlmostEqual(v[1],-1.)
        self.assertAlmostEqual(v[2],1.)

    def test_rotateFromAxisAngle(self):
        from math import pi

        v = Vec3(1.,1.,1.)
        v.rotateFromAxisAngle([1.,0.,0.],pi/2.)

        self.assertEqual(v[0],1.)
        self.assertEqual(math.floor(v[1]),-1.)
        self.assertEqual(v[2],1.)

## STATIC METHODS

    def test_dot(self):
        v1 = Vec3(1.,1.,1.)
        v2 = Vec3(1.,2.,3.)
        self.assertEqual(Vec3.dot(v1,v2),6)

    def test_cross(self):
        v = Vec3(1.,1.,1.)
        u = Vec3(1.,2.,3.)
        self.assertEqual(Vec3.cross(v,u),[1.,-2.,1.])
        self.assertEqual(Vec3.cross(u,v),[-1.,2.,-1.])


if __name__ == '__main__':
    unittest.main()
