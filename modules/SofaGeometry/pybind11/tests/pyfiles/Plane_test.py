import unittest
from SofaGeometry import Plane, Vec3, Ray

class Plane_test(unittest.TestCase):
    def test_PlaneConstructors(self):
        p = Plane()
        self.assertAlmostEqual( (p.normal-Vec3(1,0,0)).norm(), 0.0 )
        self.assertAlmostEqual( p.distance, 0.0 )

        p = Plane(Vec3(1.0,2.0,3.0), 2.0)
        self.assertAlmostEqual( (p.normal-Vec3(1.0,2.0,3.0)).norm(), 0.0 )
        self.assertAlmostEqual( p.distance, 2.0 )

        p = Plane([1.0,2.0,3.0], 3.0)
        self.assertAlmostEqual( (p.normal-Vec3(1.0,2.0,3.0)).norm(), 0.0 )
        self.assertAlmostEqual( p.distance, 3.0 )
        
    def test_PlaneRayCast(self):
        p = Plane([1.0,0.0,0.0], 0.0)
        r = Ray(origin=Vec3(-10,0,0), direction=Vec3(1,0,0))
        r = p.raycast(r)
        self.assertAlmostEqual( r, 10.0 )

        p = Plane([2.0,0.0,0.0], 0.0)
        r = Ray(origin=Vec3(-10,0,0), direction=Vec3(1,0,0))
        r = p.raycast(r)
        self.assertAlmostEqual( r, 10.0 )

        p = Plane(Vec3(1.0,0.0,0.0), Vec3(1.0,0.0,0.0))
        r = Ray(origin=Vec3(-10,0,0), direction=Vec3(1,0,0))
        r=p.raycast(r)
        self.assertAlmostEqual( r, 11.0 )

        p = Plane(Vec3(1.0,0.0,0.0), Vec3(-1.0,0.0,0.0))
        r = Ray(origin=Vec3(-10,0,0), direction=Vec3(1,0,0))
        r=p.raycast(r)
        self.assertAlmostEqual( r, 9.0 )

        p = Plane(Vec3(5.0,0.0,0.0), Vec3(-1.0,0.0,0.0))
        r = Ray(origin=Vec3(-10,0,0), direction=Vec3(1,0,0))
        l = p.raycast(r)
        p = r.getPoint(l)
        self.assertAlmostEqual( l, 9.0 )



def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Plane_test)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
