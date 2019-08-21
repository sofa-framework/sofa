import numpy
import math

class Matrix(numpy.ndarray):
    """ The Matrix class implements the following:

    Public methods:
    M = Matrix() # several constructors are implemented

    v.getSize()
    v.getNbRow()
    v.getNbCol()
    v.getTranspose()

    Static methods:
    v = Matrix.identity(n)
    """

    def __new__(cls, *args):
        """ Matrix constructors.

        Examples:

        >>> M = Matrix()
        >>> print(M)
        [[0.]]
        >>> M = Matrix(2,3)
        >>> print(M)
        [[0.,0.,0.]
        [0.,0.,0.]]
        >>> M = Matrix(3,2,1.)
        >>> print(M)
        [[1.,1.]
        [1.,1.]
        [1.,1.]]
        >>> M = Matrix(2,2,[1.,2.,3.,4.])
        >>> print(M)
        [[1.,2.]
        [3.,4.]]
        """
        if len(args)==0:
            return super(Matrix,cls).__new__(cls, shape=(1,1), dtype=float, buffer=numpy.array([[0.]]))
        if len(args)==2:
            return super(Matrix,cls).__new__(cls, shape=(args[0],args[1]), dtype=float, buffer=numpy.array([[0.]*args[0]]*args[1]))
        if len(args)==3:
            if hasattr(args[2],"__len__"):
                n = args[0]
                m = args[1]
                M = Matrix(n,m)
                if len(args[2]) == n*m:
                    for i in range(n):
                        for j in range(m):
                            M[i][j] = args[2][i*m+j]
                    return M 
            return super(Matrix,cls).__new__(cls, shape=(args[0],args[1]), dtype=type(args[2]), buffer=numpy.array([[args[2]]*args[0]]*args[1]))

        print(cls.__new__.__doc__)
        return super(Matrix,cls).__new__(cls, shape=(1,1), dtype=float, buffer=numpy.array([[0.]]))


    # Eulalie: I'm not sure about keeping that... useful for tests...
    def __eq__(self, other):
        """ Matrix overriding of __eq__ so that (M1==M2) returns a boolean.
        """

        if not hasattr(other,"__len__"):
            return False

        if (self.shape[0] != len(other)):
            return False

        if not hasattr(other[0],"__len__"):
            return False

        for i in range(self.shape[0]):
            if (self.shape[1] != len(other[i])):
                return False

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i][j] != other[i][j]:
                    return False
        return True


    def __ne__(self, other):
        """ Matrix overriding of __ne__ so that (M1!=M2) returns a boolean.
        """
        return not (self == other)


    def getSize(self):
        """ Returns the size of the matrix.

        Example:

        >>> M = Matrix(2,6)
        >>> M.getSize()
        [2,6]
        """
        return numpy.array(self.shape)


    def getNbRow(self):
        """ Returns the number of rows of the matrix.

        Example:

        >>> M = Matrix(2,6)
        >>> M.getNbRow()
        2
        """
        return self.shape[0]


    def getNbCol(self):
        """ Returns the number of columns of the matrix.

        Example:

        >>> M = Matrix(2,6)
        >>> M.getNbCol()
        6
        """
        return self.shape[1]


    def getTranspose(self):
        """ Returns the transpose matrix.

        Example:

        >>> M = Matrix([[1.,2.],[3.,4.]])
        >>> M.getTranspose()
        [[1.,3.]
        [2.,4.]]
        """
        n=self.getNbRow()
        m=self.getNbCol()
        Mt = Matrix(n,m)
        for i in range(n):
            for j in range(m):
                    Mt[j][i]=self.take(i*m+j)
        return Mt


    @staticmethod
    def identity(n):
        """ Returns the identity matrix of size (n,n).

        Example:

        >>> M = Matrix.identity(2)
        >>> print(M)
        [[1.,0.]
        [0.,1.]]
        """
        I = Matrix(n,n,0.)
        for i in range(n):
            for j in range(n):
                if i==j:
                    I[i][j]=1.
        return I


    ## multMatrixMatrix...
    ## multMatrixVector...
    ## getInverse...?
    ## getDet
