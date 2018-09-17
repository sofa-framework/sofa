import unittest
from SofaTypes import *

class Quater_test(unittest.TestCase):
    def test_QuaterConstructor(self):
        # r=Ray()
        # self.assertAlmostEquals( (r.direction-Vec3d(1.0,0.0,0.0)).norm(), 0.0 )
        # self.assertAlmostEquals( (r.origin-Vec3d(0.0,0.0,0.0)).norm(), 0.0 )

        # r=Ray(direction=Vec3d(2.0,3.0,4.0), origin=Vec3d(1.0,1.0,1.0))
        # self.assertAlmostEquals( (r.direction-Vec3d(2.0,3.0,4.0)).norm(), 0.0 )
        # self.assertAlmostEquals( (r.origin-Vec3d(1.0,1.0,1.0)).norm(), 0.0 )
        pass

def run():
    suite = unittest.TestLoader().loadTestsFromTestCase(Quater_test)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()

