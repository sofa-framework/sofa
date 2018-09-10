import unittest
from SofaGeometry import Ray, Vec3d

class Ray_test(unittest.TestCase):
    def test_RayConstructor(self):
        r=Ray()
        self.assertAlmostEquals( (r.direction-Vec3d(1.0,0.0,0.0)).norm(), 0.0 )
        self.assertAlmostEquals( (r.origin-Vec3d(0.0,0.0,0.0)).norm(), 0.0 )

        r=Ray(direction=Vec3d(2.0,3.0,4.0), origin=Vec3d(1.0,1.0,1.0))
        self.assertAlmostEquals( (r.direction-Vec3d(2.0,3.0,4.0)).norm(), 0.0 )
        self.assertAlmostEquals( (r.origin-Vec3d(1.0,1.0,1.0)).norm(), 0.0 )

    def test_RayAttrChange(self):
        r=Ray(direction=Vec3d(1.0,1.0,1.0), origin=Vec3d(0.0,0.0,0.0))
        r.direction = Vec3d(2.0,2.0,2.0)
        self.assertAlmostEquals( (r.direction-Vec3d(2.0,2.0,2.0)).norm(), 0.0 )

    def test_RayGetPoint(self):
        r=Ray(direction=Vec3d(1.0,1.0,1.0), origin=Vec3d(0.0,0.0,0.0))
        v = r.getPoint(distance=5)
        self.assertAlmostEquals( (v-Vec3d(5.0,5.0,5.0)).norm(), 0.0 )

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Ray_test)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()

