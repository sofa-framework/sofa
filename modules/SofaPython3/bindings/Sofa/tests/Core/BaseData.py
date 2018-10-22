# coding: utf8

import unittest
import numpy
import Sofa

#with obj.position.writeable(RGBAColor) as w:

class RGBAColor(numpy.ndarray):
    def __new__(cls, input_array=None):
           if input_array is None:
               obj = super(Vec4d, cls).__new__(cls, shape=(4), dtype=float)
               return obj

           if isinstance(input_array, Sofa.Core.DataContainer):
               cls.owner = input_array
               input_array = input_array.toarray()

           if input_array.ndim != 1:
               raise TypeError("Invalid dimension, expecting a 1D array, got "+str(input_array.ndim)+"D")

           # Input array is an already formed ndarray instance
           # We first cast to be our class type
           obj = numpy.asarray(input_array).view(cls)

           # Finally, we must return the newly created object:
           return obj

    def r(self):
        return self[0]

    def g(self):
        return self[1]

    def b(self):
        return self[2]

    def a(self):
        return self[3]

class Vec4d(numpy.ndarray):
    def __new__(cls, input_array=None):
           if input_array is None:
               obj = super(Vec4d, cls).__new__(cls, shape=(4), dtype=float)
               return obj

           if input_array.ndim != 1:
               raise TypeError("Invalid dimension, expecting a 1D array, got "+str(input_array.ndim)+"D")

           # Input array is an already formed ndarray instance
           # We first cast to be our class type
           obj = numpy.asarray(input_array).view(cls)

           # Finally, we must return the newly created object:
           return obj

    def x(self):
        return self[0]

    def y(self):
        return self[1]

    def z(self):
        return self[2]

    def w(self):
        return self[3]

class Test(unittest.TestCase):
        #@unittest.skip  # no reason needed
        def test_ValidDataAccess(self):
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=[[0,0,0],[1,1,1],[2,2,2]])
                self.assertTrue(c.position is not None)
        #@unittest.skip  # no reason needed
        def test_InvalidDataAccess(self):
                root = Sofa.Node("rootNode")
                self.assertRaises(AttributeError, getattr, root, "invalidData")

        #@unittest.skip  # no reason needed
        def test_DataAsArray2D(self):
                root = Sofa.Node("rootNode")
                v=[[0,0,0],[1,1,1],[2,2,2]]
                c = root.createObject("MechanicalObject", name="t", position=v)
                self.assertEqual(len(c.position), 3)
                self.assertSequenceEqual(list(c.position[0]), v[0])
                self.assertSequenceEqual(list(c.position[1]), v[1])
                self.assertSequenceEqual(list(c.position[2]), v[2])

        #@unittest.skip  # no reason needed
        def test_DataArray2DOperationInPlace(self):
                root = Sofa.Node("rootNode")
                v=numpy.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])
                c = root.createObject("MechanicalObject", name="t", position=v.tolist())
                c.position *= 2.0
                numpy.testing.assert_array_equal(c.position.toarray(), v*2.0)
                c.position += 3.0
                numpy.testing.assert_array_equal(c.position.toarray(), (v*2.0)+3.0)

        #@unittest.skip  # no reason needed
        def test_DataArray2DSetFromList(self):
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=v)
                c.position = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
                numpy.testing.assert_array_equal(c.position.toarray(), [[1.0,1.0,1.0],[2.0,2.0,2.0],[3.0,3.0,3.0],[4.0,4.0,4.0]])

        #@unittest.skip  # no reason needed
        def test_DataArray2DResizeFromArray(self):
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=v)
                zeros = numpy.zeros((100,3), dtype=numpy.float64)
                c.position = zeros
                numpy.testing.assert_array_equal(c.position.toarray(), zeros)

        #@unittest.skip  # no reason needed
        def test_DataArray2DInvalidResizeFromArray(self):
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=v)
                zeros = numpy.zeros((4,100), dtype=numpy.float64)
                def d():
                    c.position = zeros
                self.assertRaises(IndexError, d)

        #@unittest.skip  # no reason needed
        def test_DataArray2DSetFromArray(self):
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=v)

                zeros = numpy.zeros((500,3), dtype=numpy.float64)
                c.position = zeros
                numpy.testing.assert_array_equal(c.position.toarray(), zeros)

                ones = numpy.ones((1000,3), dtype=numpy.float32)
                c.position = ones
                numpy.testing.assert_array_equal(c.position.toarray(), ones)

                zeros = numpy.zeros((500,3), dtype=numpy.float32)
                c.position = zeros
                numpy.testing.assert_array_equal(c.position.toarray(), zeros)

        @unittest.skip  # no reason needed
        def test_DataArray2DElementWiseOperation(self):
                root = Sofa.Node("rootNode")
                m=[[1,0,0],[0,1,0],[0,0,1]]
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                c = root.createObject("MechanicalObject", name="t", position=v)
                c.position *= c.position

        def test_DataArrayCreateFromNumpy(self):
                root = Sofa.Node("rootNode")
                v=numpy.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])
                c = root.createObject("MechanicalObject", name="t", position=v)


        #@unittest.skip  # no reason needed
        def test_DataArray2DOperation(self):
                root = Sofa.Node("rootNode")
                v=numpy.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])
                c = root.createObject("MechanicalObject", name="t", position=v.tolist())

                c2 = c.position * 2.0
                numpy.testing.assert_array_equal(c.position.toarray(), v)
                numpy.testing.assert_array_equal(c2, v*2.0)

                c2 = c.position + 2.0
                numpy.testing.assert_array_equal(c.position.toarray(), v)
                numpy.testing.assert_array_equal(c2, v+2.0)

        #@unittest.skip  # no reason needed
        def test_DataAsArray1D(self):
                root = Sofa.Node("rootNode")
                v=[[0,0,0],[1,1,1],[2,2,2]]
                c = root.createObject("MechanicalObject", name="t", position=v)
                self.assertEqual(len(c.showColor), 4)

        #@unittest.skip  # no reason needed
        def test_DataWrapper1D(self):
                root = Sofa.Node("rootNode")
                v=[[0,0,0],[1,1,1],[2,2,2]]
                root.createObject("MechanicalObject", name="obj", position=v)

                c = RGBAColor(root.obj.showColor)
                self.assertEqual(c.r(), 1.0)

                with root.obj.showColor.writeable(RGBAColor) as color:
                    color[0] = 2.0

                self.assertEqual(color.r(), 2.0)
                def t(c):
                    c[0]=1.0
                self.assertRaises(ValueError, (lambda c: t(c)), color )

        def test_DataAsContainerNumpyArray(self):
                root = Sofa.Node("rootNode")
                v=numpy.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])
                c = root.createObject("MechanicalObject", name="t", position=v.tolist())

                with c.position.writeable() as wa:
                    self.assertEqual(wa.shape, (4,3))
                    self.assertEqual(wa[0,0], 0.0)
                    self.assertEqual(wa[1,1], 1.0)
                    self.assertEqual(wa[2,2], 2.0)
                    numpy.testing.assert_array_equal(c.position.toarray(), v)

        def test_DataAsContainerNumpyArrayRepeat(self):
               root = Sofa.Node("rootNode")
               v=numpy.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3]])
               c = root.createObject("MechanicalObject", name="t", position=v.tolist())

               with c.position.writeable() as wa:
                   wa *= 2.0
                   self.assertEqual(wa.shape, (4,3))
                   self.assertEqual(wa[0,0], 0.0)
                   self.assertEqual(wa[1,1], 2.0)
                   self.assertEqual(wa[2,2], 4.0)
                   numpy.testing.assert_array_equal(wa, v*2.0)

               ### Checks that the data are correctly changed in the writeable block
               numpy.testing.assert_array_equal(c.position.toarray(), v*2.0)

               ### Checks that trying to access wa object in write correctly raise
               ### an error.
               def t(c):
                   c[0,0]=1.0
               self.assertRaises(ValueError, (lambda c: t(c)), wa )

               ### Checks that the previously defined blocks is correctly re-created.
               with c.position.writeable() as wa:
                   wa *= 2.0
                   self.assertEqual(wa.shape, (4,3))
                   self.assertEqual(wa[0,0], 0.0)
                   self.assertEqual(wa[1,1], 4.0)
                   self.assertEqual(wa[2,2], 8.0)
                   numpy.testing.assert_array_equal(wa, v*4.0)

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

def createScene(rootNode):
        runTests()
