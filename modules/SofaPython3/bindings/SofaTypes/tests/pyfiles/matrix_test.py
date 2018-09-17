import unittest
from SofaGeometry import Plane, Vec3d, Ray

class Matrix_test(unittest.TestCase):
    def test_MatrixConstructors(self):
        # p = Plane()
        # self.assertAlmostEqual( (p.direction-Vec3d(1,0,0)).norm(), 0.0 )
        # self.assertAlmostEqual( p.distance, 0.0 )

        # p = Plane(Vec3d(1.0,2.0,3.0), 2.0)
        # self.assertAlmostEqual( (p.direction-Vec3d(1.0,2.0,3.0)).norm(), 0.0 )
        # self.assertAlmostEqual( p.distance, 2.0 )

        # p = Plane([1.0,2.0,3.0], 3.0)
        # self.assertAlmostEqual( (p.direction-Vec3d(1.0,2.0,3.0)).norm(), 0.0 )
        # self.assertAlmostEqual( p.distance, 3.0 )
        pass


def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Matrix_test)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
