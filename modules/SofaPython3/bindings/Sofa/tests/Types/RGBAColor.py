# coding: utf8

import Sofa
import unittest

class Test(unittest.TestCase):
    def test_constructor_default(self):
        v0 = Sofa.Types.RGBAColor()
        self.assertEqual( v0.r(), 0 )
        self.assertEqual( v0.g(), 0 )
        self.assertEqual( v0.b(), 0 )
        self.assertEqual( v0.a(), 0 )

    def test_constructor_fromList(self):
        v0 = Sofa.Types.RGBAColor([1.0,2.0,3.0,4.0])
        self.assertEqual( v0.r(), 1.0 )
        self.assertEqual( v0.g(), 2.0 )
        self.assertEqual( v0.b(), 3.0 )
        self.assertEqual( v0.a(), 4.0 )

    def test_constructor_fromInvalidList(self):
        self.assertRaises(ValueError, Sofa.Types.RGBAColor, [1.0,2.0,3.0])
        self.assertRaises(ValueError, Sofa.Types.RGBAColor, [1.0,2.0,3.0,10,100])
        self.assertRaises(TypeError, Sofa.Types.RGBAColor, [1.0,2.0,"three",4.0])


def getTestsName():
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    return [ test.id().split(".")[2] for test in suite]

def runTests():
        import sys
        suite = None
        if( len(sys.argv) == 1 ):
            suite = unittest.TestLoader().loadTestsFromTestCase(Test)
        else:
            suite = unittest.TestSuite()
            suite.addTest(Test(sys.argv[1]))
        return unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()
