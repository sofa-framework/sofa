import unittest
from matrix import *

class Matrix_test(unittest.TestCase):

    def test_equal(self):
        M1 = Matrix(2,3)
        M2 = Matrix(2,3,1.5)
        M3 = Matrix(2,4)

        self.assertEqual(M1,M1)
        self.assertNotEqual(M1,2.)
        self.assertNotEqual(M1,[[2.,0.,1.],[1.]])
        self.assertNotEqual(M1,M3)
        self.assertNotEqual(M1,M2)

    def test_constructors(self):
        M = Matrix()
        self.assertEqual(M,[[0.]])

        M = Matrix(2,3)
        self.assertEqual(M,[[0.,0.,0.],[0.,0.,0.]])

        M = Matrix(3,2,1.)
        self.assertEqual(M,[[1.,1.],[1.,1.],[1.,1.]])

        M = Matrix(2,3,[1.,2.,3.,4.,5.,6.])
        self.assertEqual(M,[[1.,2.,3.],[4.,5.,6.]])

        M = Matrix(3,2,[1.,2.,3.,4.,5.,6.])
        self.assertEqual(M,[[1.,2.],[3.,4.],[5.,6.]])

    def test_operators(self):
        M0 = Matrix(2,3,0.)
        M1 = Matrix(2,3,1.)
        M2 = Matrix(2,3,2.5)
        M3 = Matrix(2,3,3.5)
        self.assertEqual(M1+M2,M3)
        self.assertEqual(M1-M1,M0)
        self.assertEqual(M1*2.5,M2)
        self.assertEqual(M2/2.5,M1)

    def test_getSize(self):
        M = Matrix(2,3)
        size = M.getSize()
        self.assertEqual(size[0],2)
        self.assertEqual(size[1],3)

    def test_getNbRow_getNbCol(self):
        M = Matrix(2,5)
        self.assertEqual(M.getNbRow(),2)
        self.assertEqual(M.getNbCol(),5)

    def test_getTranspose(self):
        M = Matrix(2,2,[1.,2.,3.,4.])
        self.assertEqual(M.getTranspose(),[[1.,3.],[2.,4.]])


## STATIC METHODS

    def test_identity(self):
        I = Matrix.identity(3)
        M = Matrix(3,3,1.)
        self.assertEqual(I,[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        self.assertNotEqual(I,M)



if __name__ == '__main__':
    unittest.main()
