import numpy
import math
from math import pi

class Quat(numpy.ndarray):
    """ The Quat class implements the following:

    Public methods:
    q = Quat() # several constructors are implemented

    q.rotateFromQuat(q1)
    q.rotateFromEuler(v)
    q.normalize()
    q.flip()

    q.getIm()
    q.getRe()
    q.getNorm()
    q.getInverse()
    q.getConjugate()
    q.getMatrix()
    q.getEulerAngles()
    q.getAxisAngle()
    q.toString()

    Static methods:
    q = Quat.product(q1,q2)
    q = Quat.createFromEuler([x,y,z])
    q = Quat.createFromAxisAngle([axis],angle)
    """

    def __new__(cls, *args):
        """ Quat constructor expects zero, one, or four arguments. Quat has the Sofa format i.e (x,y,z,w).

        Examples:

        >>> q = Quat()
        >>> print(q)
        [0.,0.,0.,1.]
        >>> q = Quat(0.,0.,0.,1.)
        >>> print(q)
        [0.,0.,0.,1.]
        >>> q = Quat([0.,0.,0.,1.])
        >>> print(q)
        [0.,0.,0.,1.]
        """
        if len(args)==0:
            return super(Quat,cls).__new__(cls, shape=(4,), dtype=float, buffer=numpy.array([0.,0.,0.,1.]))
        elif hasattr(args[0],"__len__") and len(args[0])==4:
            return super(Quat,cls).__new__(cls, shape=(4,), dtype=type(args[0][0]), buffer=numpy.array([args[0][0],args[0][1],args[0][2],args[0][3]]))
        elif len(args)==4:
            return super(Quat,cls).__new__(cls, shape=(4,), dtype=type(args[0]), buffer=numpy.array([args[0],args[1],args[2],args[3]]))

        print(cls.__new__.__doc__)
        return super(Quat,cls).__new__(cls, shape=(4,), dtype=float, buffer=numpy.array([0.,0.,0.,1.]))


    def __eq__(self, other):
        """ Quat overriding of __eq__ so that (q1==q2) returns a boolean.
        """
        results = (super(Quat,self).__eq__(other))
        for result in results:
            if result == False:
                return False
        return True


    def __ne__(self, other):
        """ Quat overriding of __ne__ so that (q1!=q2) returns a boolean.
        """
        return not (self == other)


    def normalize(self, *args):
        """ Function normalize of class Quat normalize the vector. The function expects no argument.
        """
        self /= self.getNorm()


    def rotateFromQuat(self, qb):
        """Function rotateFromQuat of class Quat rotates the current Quat from the given one.

        Examples:

        >>> q1 = Quat.createFromAxisAngle([1., 0., 0.], pi/2.)
        >>> q2 = Quat.createFromAxisAngle([0., -1., 0.], pi/2.)
        >>> q1.rotateFromQuat(q2)
        >>> print(q1)
        [ 0.5 -0.5 -0.5  0.5]
        """

        self.put(range(4),self.product(self,qb))


    def rotateFromEuler(self, v, axes="sxyz"):
        """Function rotateFromEuler of class Quat combine the current Quat from euler angles.

        Examples:

        >>> q = Quat.createFromAxisAngle([1., 0., 0.], pi/2.)
        >>> q.rotateFromEuler([0.,-pi/2.,0.])
        >>> print(q)
        [ 0.5 -0.5 -0.5  0.5]
        """

        q = Quat.createFromEuler(v)
        self.put(range(4),self.product(self,q))


    def flip(self):
        """Function flip of class Quat flips the quaternion to the real positive hemisphere if needed.
        """
        if self.getRe() < 0:
            self.put(range(4),-1*self)



    def getNorm(self):
        """ Returns the norm of the quaternion.
        """
        return numpy.linalg.norm(self)


    def getRe(self):
        """Returns the real part of the quaternion.

        Example:

        >>> q = Quat(0.65,0.,0.,0.75)
        >>> q.getRe()
        0.75
        """
        return float(self.take(3))


    def getIm(self):
        """Returns the imaginary part of the quaternion.

        Example:

        >>> q = Quat(0.65,0.,0.,0.75)
        >>> q.getIm()
        [0.65,0.,0.]
        """
        return numpy.array(self.take(range(3)))


    def getAxisAngle(self):
        """ Returns rotation vector corresponding to unit quaternion in the form of [axis, angle]
        """
        import sys
        q = Quat(self)
        q.flip()  # flip q first to ensure that angle is in the [-0, pi] range

        angle = 2.0* math.acos(q.getRe())

        if angle > sys.float_info.epsilon:
            return [ q.getIm() / math.sin(angle/2.), angle ]

        norm = numpy.linalg.norm(q.getIm())
        if norm > sys.float_info.epsilon:
            sign = 1.0 if angle > 0 else -1.0
            return [ q.getIm() * (sign / norm), angle ]

        return [ numpy.zeros(3), angle ]

    def getEulerAngles(self, axes='sxyz'):
        """Returns the Euler angles in radian for specified axis sequence.
        """

        M = self.getMatrix()

        try:
            firstaxis, parity, repetition, frame = AXES_TO_TUPLE[axes.lower()]
        except (AttributeError, KeyError):
            TUPLE_TO_AXES[axes]  # validation
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = NEXT_AXIS[i+parity]
        k = NEXT_AXIS[i-parity+1]

        a = numpy.empty((3, ))

        if repetition:
            sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
            if sy > EPS:
                a[0] = math.atan2( M[i, j],  M[i, k])
                a[1] = math.atan2( sy,       M[i, i])
                a[2] = math.atan2( M[j, i], -M[k, i])
            else:
                a[0] = math.atan2(-M[j, k],  M[j, j])
                a[1] = math.atan2( sy,       M[i, i])
                a[2] = 0.0
        else:
            cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
            if cy > EPS:
                a[0] = math.atan2( M[k, j],  M[k, k])
                a[1] = math.atan2(-M[k, i],  cy)
                a[2] = math.atan2( M[j, i],  M[i, i])
            else:
                a[0] = math.atan2(-M[j, k],  M[j, j])
                a[1] = math.atan2(-M[k, i],  cy)
                a[2] = 0.0

        if parity:
            a[0], a[1], a[2] = -a[0], -a[1], -a[2]
        if frame:
            a[0], a[2] = a[2], a[0]
        return a


    def getMatrix(self):
        """Returns the convertion of the quaternion into rotation matrix form.
        """
        q = Quat(self)

        # Repetitive calculations
        q44 = q[3]**2
        q12 = q[0] * q[1]
        q13 = q[0] * q[2]
        q14 = q[0] * q[3]
        q23 = q[1] * q[2]
        q24 = q[1] * q[3]
        q34 = q[2] * q[3]

        matrix = numpy.empty((3,3))

        # The diagonal
        matrix[0, 0] = 2.0 * (q[0]**2 + q44) - 1.0
        matrix[1, 1] = 2.0 * (q[1]**2 + q44) - 1.0
        matrix[2, 2] = 2.0 * (q[2]**2 + q44) - 1.0

        # Off-diagonal
        matrix[0, 1] = 2.0 * (q12 - q34)
        matrix[0, 2] = 2.0 * (q13 + q24)
        matrix[1, 2] = 2.0 * (q23 - q14)

        matrix[1, 0] = 2.0 * (q12 + q34)
        matrix[2, 0] = 2.0 * (q13 - q24)
        matrix[2, 1] = 2.0 * (q23 + q14)

        return matrix


    def getConjugate(self):
        """Returns the conjugate of the quaternion.

        Example:

        >>> q = Quat(0.707,0.,0.,0.707)
        >>> q.getConjugate()
        [-0.707,0.,0.,0.707]
        """
        return Quat(-self.take(0),-self.take(1),-self.take(2),self.take(3))


    def getInverse(self):
        """Returns the inverse of the quaternion.

        If you are dealing with unit quaternions, use getConjugate() instead.
        """
        return  self.getConjugate() / self.getNorm()**2


    def toString(self):
        """Returns the quaternion in string format.
        """
        return  str(self.take(0))+" "+str(self.take(1))+" "+str(self.take(2))+" "+str(self.take(3))


    @staticmethod
    def createFromAxisAngle(axis, angle):
        """ Function createQuatFromAxis from quat expects two arguments. Quat has the Sofa format i.e (x,y,z,w).

        Examples:

        >>> q = Quat.createQuatFromAxis([1.,0.,0.],pi/2.)
        >>> print(q)
        [0.707,0.,0.,0.707]

        Note that the angle should be in radian.
        """
        from quat import Quat
        q = Quat()
        q[0]=axis[0]*math.sin(angle/2.)
        q[1]=axis[1]*math.sin(angle/2.)
        q[2]=axis[2]*math.sin(angle/2.)
        q[3]=math.cos(angle/2.)

        q.normalize()
        return q


    @staticmethod
    def createFromEuler(a, axes='sxyz', inDegree=False):
        """Returns a quaternion from Euler angles (in radian) and axis sequence.
        The quaternion is of type Quat.

        Args:

        a is a list of three Euler angles [x,y,z]
        axes : One of 24 axis sequences as string or encoded tuple

        Example:

        >>> q = Quat.createFromEuler([-pi, 0., 0.], 'sxyz')
        >>> print(q)
        [ 1.0 0.0  0.0  0.0]

        >>> q = Quat.createFromEuler([-pi/2., pi/2., 0.], 'ryxz') #r stands for repetition
        >>> print(q)
        [ 0.5 -0.5  0.5  0.5]
        """

        if inDegree:
            a = [a[0]*pi/180, a[1]*pi/180, a[2]*pi/180]

        try:
            firstaxis, parity, repetition, frame = AXES_TO_TUPLE[axes.lower()]
        except (AttributeError, KeyError):
            TUPLE_TO_AXES[axes]  # validation
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = NEXT_AXIS[i+parity]
        k = NEXT_AXIS[i-parity+1]

        if frame:
            a[0], a[2] = a[2], a[0]
        if parity:
            a[1] = -a[1]

        a[0] /= 2.0
        a[1] /= 2.0
        a[2] /= 2.0
        ci = math.cos(a[0])
        si = math.sin(a[0])
        cj = math.cos(a[1])
        sj = math.sin(a[1])
        ck = math.cos(a[2])
        sk = math.sin(a[2])
        cc = ci*ck
        cs = ci*sk
        sc = si*ck
        ss = si*sk

        q = Quat()
        if repetition:
            q[3] = cj*(cc - ss)
            q[i] = cj*(cs + sc)
            q[j] = sj*(cc + ss)
            q[k] = sj*(cs - sc)
        else:
            q[3] = cj*cc + sj*ss
            q[i] = cj*sc - sj*cs
            q[j] = cj*ss + sj*cc
            q[k] = cj*cs - sj*sc
        if parity:
            q[j] *= -1.0

        return q


    @staticmethod
    def product(qa, qb):
        """Use this product to compose the rotations represented by two quaterions.

        Example:

        >>> q1 = Quat()
        >>> q2 = Quat()
        >>> Quat.product(q1,q2)
        [0.,0.,0.,1.]
        """

        # Here is a readable version :
        # array([ qa[3]*qb[0] + qb[3]*qa[0] + qa[1]*qb[2] - qa[2]*qb[1],
        # qa[3]*qb[1] + qb[3]*qa[1] + qa[2]*qb[0] - qa[0]*qb[2],
        # qa[3]*qb[2] + qb[3]*qa[2] + qa[0]*qb[1] - qa[1]*qb[0],
        # qa[3]*qb[3] - qb[0]*qa[0] - qa[1]*qb[1] - qa[2]*qb[2] ])
        return Quat(numpy.hstack( (qa.getRe()*qb.getIm() + qb.getRe()*qa.getIm() + numpy.cross( qa.getIm(), qb.getIm() ), [qa.getRe() * qb.getRe() - numpy.dot( qa.getIm(), qb.getIm())] )))


##### adapted from http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

# epsilon for testing whether a number is close to zero
EPS = numpy.finfo(float).eps * 4.0

# Axis sequences for Euler angles
NEXT_AXIS = [1, 2, 0, 1]

# Map axes strings to/from tuples of inner axis, parity, repetition, frame
AXES_TO_TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

TUPLE_TO_AXES = dict((v, k) for k, v in AXES_TO_TUPLE.items())
