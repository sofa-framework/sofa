import unittest
from math import sin, cos, pi
from quat import *

class Quat_test(unittest.TestCase):


    def test_equal(self):
        q1 = Quat()
        q2 = Quat(0.707,0.,0.,0.707)
        self.assertEqual(q1,q1)
        self.assertNotEqual(q1,q2)

        self.assertEqual(q1,[0.,0.,0.,1.])
        self.assertNotEqual(q1,[0.707,0.,0.,0.707])


    def test_constructors(self):
        q = Quat()
        self.assertEqual(q,[0.,0.,0.,1.])

        q = Quat(0.707,0.,0.,0.707)
        self.assertEqual(q,[0.707,0.,0.,0.707])

        q = Quat([0.707,0.,0.,0.707])
        self.assertEqual(q,[0.707,0.,0.,0.707])


## PUBLICS METHODS


    def test_getNorm(self):
        q = Quat()
        self.assertEqual(q.getNorm(), 1.)


    def test_normalize(self):
        q = Quat(1.,0.,2.,2.)
        q.normalize()
        self.assertEqual(q, [1./3.,0.,2./3.,2./3.])


    def test_realPart(self):
        q = Quat()
        self.assertEqual(q.getRe(), 1.)


    def test_imaginaryPart(self):
        q = Quat(1.,2.,3.,4.)
        self.assertEqual(q.getIm()[0], 1.)
        self.assertEqual(q.getIm()[1], 2.)
        self.assertEqual(q.getIm()[2], 3.)


    def test_flip(self):
        q = Quat(0.707,0.,0.,-0.707)
        q.flip()
        self.assertEqual(q, [-0.707,0.,0.,0.707])



    def test_conjugate(self):
        q = Quat()
        self.assertEqual(q.getConjugate(), q)

        q = Quat(0.5,0.5,0.5,0.5)
        self.assertEqual(q.getConjugate(), [-0.5,-0.5,-0.5,0.5])


    def test_getInverse(self):
        q = Quat()
        self.assertEqual(q.getInverse(), q)

        q = Quat(0.5,0.5,0.5,0.5)
        self.assertEqual(q.getInverse(), q.getConjugate())

        q = Quat(1.,1.,1.,1.)
        self.assertEqual(q.getInverse(), [-0.25,-0.25,-0.25,0.25])


    def test_getAxisAngle(self):
        q = Quat.createFromAxisAngle([1.,0.,0.],pi/3.)
        results = q.getAxisAngle()
        self.assertAlmostEqual(results[0][0], 1.)
        self.assertEqual(results[0][1], 0.)
        self.assertEqual(results[0][2], 0.)
        self.assertAlmostEqual(results[1], pi/3.)


    def test_getEulerAngles(self):
        q = Quat.createFromEuler([-pi/4.,0.,0.])
        e = q.getEulerAngles()
        self.assertAlmostEqual(e[0], -pi/4.)
        self.assertEqual(e[1], 0.)
        self.assertEqual(e[2], 0.)


## STATIC METHODS


    def test_createFromAxisAngle(self):
        q = Quat.createFromAxisAngle([1.,0.,0.],pi/2.)
        self.assertEqual(q, [sin(pi/4.),0.,0.,cos(pi/4.)])


    def test_createFromEuler(self):
        q = Quat.createFromEuler([pi/2.,0.,0.])
        self.assertEqual(q, [sin(pi/4.),0.,0.,cos(pi/4.)])
        q = Quat.createFromEuler([90.,0.,0.], inDegree=True)
        self.assertEqual(q, [sin(pi/4.),0.,0.,cos(pi/4.)])
        q = Quat.createFromEuler([0.,-pi/2.,0.])
        self.assertEqual(q, [0.,-sin(pi/4.),0.,cos(pi/4.)])
        q = Quat.createFromEuler([0.,pi/2.,0.],"syxz")
        self.assertEqual(q, [sin(pi/4.),0.,0.,cos(pi/4.)])


    def test_createFromEuler_against_rotateFromQuat(self):
        q1 = Quat.createFromEuler([pi/2.,-pi/2.,0.],"rxyz")
        q2 = Quat.createFromEuler([pi/2.,0.,0.],"sxyz")
        q3 = Quat.createFromEuler([0.,-pi/2.,0.],"sxyz")
        q2.rotateFromQuat(q3)
        self.assertEqual(q1,q2)

        q1 = Quat.createFromEuler([-pi/2.,pi/2.,0.],"ryxz")
        q2 = Quat.createFromEuler([pi/2.,0.,0.],"sxyz")
        q3 = Quat.createFromEuler([0.,-pi/2.,0.],"sxyz")
        q3.rotateFromQuat(q2)
        self.assertEqual(q1,q3)

        q1 = Quat.createFromEuler([pi/2.,-pi/2.,pi/2.],"rxyz")
        q2 = Quat.createFromEuler([pi/2.,0.,0.],"sxyz")
        q3 = Quat.createFromEuler([0.,-pi/2.,0.],"sxyz")
        q4 = Quat.createFromEuler([0.,0.,pi/2.],"sxyz")
        q2.rotateFromQuat(q3)
        q2.rotateFromQuat(q4)
        self.assertEqual(q1,q2)


    def test_rotateFromEuler(self):
        q = Quat.createFromAxisAngle([1., 0., 0.], pi/2.)
        q.rotateFromEuler([0.,-pi/2.,0.])

        r = [0.5,-0.5,-0.5,0.5]
        for i in range(4):
            self.assertAlmostEqual(q[i],r[i])


if __name__ == '__main__':
    unittest.main()
