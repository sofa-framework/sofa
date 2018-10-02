import unittest
import numpy
import Sofa

class RGBAColor(numpy.ndarray):
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
        def test_ValidDataAccess(self):
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=[[0,0,0],[1,1,1],[2,2,2]])
                self.assertTrue(c.position is not None)
                
        def test_InvalidDataAccess(self):
                root = Sofa.Node("rootNode")
                self.assertRaises(AttributeError, getattr, root, "invalidData")

        def test_DataAsArray2D(self):
                root = Sofa.Node("rootNode")
                v=[[0,0,0],[1,1,1],[2,2,2]]
                c = root.createObject("MechanicalObject", name="t", position=v)
                self.assertEqual(len(c.position), 3)
                self.assertSequenceEqual(list(c.position[0]), v[0])
                self.assertSequenceEqual(list(c.position[1]), v[1])
                self.assertSequenceEqual(list(c.position[2]), v[2])

        def test_DataArray2DOperationInPlace(self):
                root = Sofa.Node("rootNode")
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                c = root.createObject("MechanicalObject", name="t", position=v)

                c.position *= 2.0
                self.assertSequenceEqual(list(c.position[0]), [0.0,0.0,0.0])
                self.assertSequenceEqual(list(c.position[1]), [2.0,2.0,2.0])
                self.assertSequenceEqual(list(c.position[2]), [4.0,4.0,4.0])
                self.assertSequenceEqual(list(c.position[3]), [6.0,6.0,6.0])
                c.position += 1.0
                self.assertSequenceEqual(list(c.position[0]), [1.0,1.0,1.0])
                self.assertSequenceEqual(list(c.position[1]), [3.0,3.0,3.0])
                self.assertSequenceEqual(list(c.position[2]), [5.0,5.0,5.0])
                self.assertSequenceEqual(list(c.position[3]), [7.0,7.0,7.0])

        def test_DataArray2DSetFromList(self):
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=v)
                c.position = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
                numpy.testing.assert_array_equal(c.position, [[1.0,1.0,1.0],[2.0,2.0,2.0],[3.0,3.0,3.0],[4.0,4.0,4.0]])

        def test_DataArray2DResizeFromArray(self):
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=v)
                zeros = numpy.zeros((100,3), dtype=numpy.float64)
                c.position = zeros
                numpy.testing.assert_array_equal(c.position, zeros)

        def test_DataArray2DInvalidResizeFromArray(self):
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=v)
                zeros = numpy.zeros((4,100), dtype=numpy.float64)
                def d():
                    c.position = zeros
                self.assertRaises(IndexError, d)

        def test_DataArray2DSetFromArray(self):
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=v)
                zeros = numpy.zeros((4,3), dtype=numpy.float64)
                c.position = zeros
                numpy.testing.assert_array_equal(c.position, zeros)

                zeros = numpy.zeros((4,3), dtype=numpy.float32)
                c.position = zeros
                numpy.testing.assert_array_equal(c.position, zeros)

                zeros = numpy.ones((4,3), dtype=numpy.float32)
                c.position = zeros
                numpy.testing.assert_array_equal(c.position, zeros)

                zeros = numpy.ones((4,3), dtype=numpy.float64)
                c.position = zeros
                numpy.testing.assert_array_equal(c.position, zeros)

        def test_DataArray2DElementWiseOperation(self):
                root = Sofa.Node("rootNode")
                m=[[1,0,0],[0,1,0],[0,0,1]]
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                c = root.createObject("MechanicalObject", name="t", position=v)
                c.position *= c.position

        def test_DataArray2DOperation(self):
                root = Sofa.Node("rootNode")
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                c = root.createObject("MechanicalObject", name="t", position=v)

                c2 = c.position * 2.0
                self.assertSequenceEqual(list(c.position[0]), [0.0,0.0,0.0])
                self.assertSequenceEqual(list(c.position[1]), [1.0,1.0,1.0])
                self.assertSequenceEqual(list(c.position[2]), [2.0,2.0,2.0])
                self.assertSequenceEqual(list(c.position[3]), [3.0,3.0,3.0])

                self.assertSequenceEqual(list(c2[0]), [0.0,0.0,0.0])
                self.assertSequenceEqual(list(c2[1]), [2.0,2.0,2.0])
                self.assertSequenceEqual(list(c2[2]), [4.0,4.0,4.0])
                self.assertSequenceEqual(list(c2[3]), [6.0,6.0,6.0])

        def test_DataAsArray1D(self):
                root = Sofa.Node("rootNode")
                v=[[0,0,0],[1,1,1],[2,2,2]]
                c = root.createObject("MechanicalObject", name="t", position=v)
                self.assertEqual(len(c.showColor), 4)

        def test_DataAsContainerMemoryView(self):
                root = Sofa.Node("rootNode")
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                c = root.createObject("MechanicalObject", name="t", position=v)                
                m = memoryview(c.position)           
                self.assertEqual(m.shape, (4,3))
                self.assertEqual(m[0,0], 0.0)
                self.assertEqual(m[1,1], 1.0)
                self.assertEqual(m[2,2], 2.0)
                self.assertEqual(m.tolist(), v)
                       
def runTests():
        suite = unittest.TestLoader().loadTestsFromTestCase(Test)
        return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()

def createScene(rootNode):
        runTests()
