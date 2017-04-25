import numpy as np
import math
import sys
from contextlib import contextmanager

from numpy.linalg import norm

GL = None

def vec(*coords):
    return np.array(coords)

deg = 180.0 / math.pi

ex = vec(1, 0, 0)
ey = vec(0, 1, 0)
ez = vec(0, 0, 1)

# slices for quaternion/rigid/deriv
imag_slice = slice(None, 3)
real_index = -1

angular_slice = slice(3, None)
linear_slice = slice(None, 3)

orient_slice = slice(3, None)
center_slice = slice(None, 3)


class Rigid3(np.ndarray):

    dim = 6
    __slots__ = ()

    
    class Deriv(np.ndarray):
        '''lie algebra element as (translation, rotation)'''
        __slots__ = ()


        def __new__(cls, *args, **kwargs):
            return np.ndarray.__new__(cls, 6)
        
        @property
        def linear(self):
            return self[linear_slice].view( np.ndarray )

        @linear.setter
        def linear(self, value):
            self[linear_slice] = value

        @property
        def angular(self):
            return self[angular_slice].view( np.ndarray )

        @angular.setter
        def angular(self, value):
            self[angular_slice] = value


        def exp(self):
            '''SE(3) exponential'''
            res = Rigid3()

            res.orient = Quaternion.exp( x.angular )
            res.center = res.orient( Quaternion.dexp( x.angular ).dot( x.linear ) )

            return res

            
    @property
    def center(self):
        return self[center_slice].view( np.ndarray )

    @center.setter
    def center(self, value):
        self[center_slice] = value

    @property
    def orient(self):
        return self[orient_slice].view(Quaternion)

    @orient.setter
    def orient(self, value):
        self[orient_slice] = value

    def __new__(cls, *args):
        return np.ndarray.__new__(cls, 7)
        
    def __init__(self, value = None, **kwargs):
        '''construct a rigid transform from given value, identity if none'''
        if value is None:
            self[-1] = 1
            self[:6] = 0
        else:
            self[:] = value

        for k, v in kwargs.iteritems(): setattr(self, k, v)

    def inv(self):
        '''SE(3) inverse'''
        res = Rigid3()
        res.orient = self.orient.inv()
        res.center = -res.orient(self.center)
        return res

    def __mul__(self, other):
        '''SE(3) product'''
        
        res = Rigid3()

        res.orient = self.orient * other.orient
        res.center = self.center + self.orient(other.center)
        
        return res

    def __call__(self, x):
        '''applies rigid transform to vector x'''
        return self.center + self.orient(x)


    def Ad(self):
        '''SE(3) group adjoint matrix in lie algebra coordinates'''
        res = np.zeros((6, 6))

        R = self.orient.matrix()
        t = Quaternion.hat(self.center)

        res[angular_slice, angular_slice] = R
        res[linear_slice, linear_slice] = R

        res[linear_slice, angular_slice] = t.dot(R)

        return res

    def matrix(self):
        '''homogeneous matrix for rigid transformation'''

        res = np.zeros( (4, 4) )

        res[:3, :3] = self.orient.matrix()
        res[:3, 3] = self.center

        res[3, 3] = 1

        return res

    
    def log(self):
        '''SE(3) logarithm'''
        res = Rigid3.Deriv()

        res.angular = self.orient.log()
        res.linear = self.orient.dlog().dot( self.orient.conj()( self.center ) )

        return res


    @contextmanager
    def gl_frame(self):
        '''draw in the current rigid frame'''

        global GL
        if GL is None: from OpenGL import GL
        
        GL.glPushMatrix()
        try:
            GL.glTranslate(*self.center)
            axis, angle = self.orient.axis_angle()
            if angle:
                GL.glRotate(angle * deg, *axis)
            yield
        finally:
            GL.glPopMatrix()


        
class Quaternion(np.ndarray):

    __slots__ = ()
    
    dim = 3
    epsilon = sys.float_info.epsilon
    
    def __new__(cls, *args):
        return np.ndarray.__new__(cls, 4)
        
    def __init__(self, value = None):
        '''construct a quaternion with given values, identity by default'''
        if value is None:
            self.real = 1
            self.imag = 0
        else:
            self[:] = value
            
    def inv(self):
        '''quaternion inverse. use conj for unit quaternions instead'''
        return self.conj() / self.dot(self)
    
    def conj(self):
        '''quaternion conjugate'''
        res = Quaternion()
        res.real = self.real
        res.imag = -self.imag

        return res

    @property
    def real(self):
        '''quaternion real part'''
        return self[real_index]

    @real.setter
    def real(self, value): self[real_index] = value

    @property
    def imag(self):
        '''quaternion imaginary part'''
        return self[imag_slice].view( np.ndarray )

    @imag.setter
    def imag(self, value): self[imag_slice] = value
    
    def normalize(self):
        '''normalize quaternion'''
        self /= norm(self)

    def flip(self):
        '''flip quaternion in the real positive halfplane, if needed'''
        if self.real < 0: self = -self

    def __mul__(self, other):
        '''quaternion product'''
        res = Quaternion()

        # TODO there might be a more efficient way                
        res.real = self.real * other.real - self.imag.dot(other.imag)
        res.imag = self.real * other.imag + other.real * self.imag + np.cross(self.imag, other.imag)
        
        return res
         

    def __call__(self, x):
        '''rotate a vector. self should be normalized'''
        
        # euler-rodrigues formula
        cross = np.cross(self.imag, 2 * x)
        return x + self.real * cross + np.cross(self.imag, cross)


    def matrix(self):
        '''rotation matrix conversion'''

        K = Quaternion.hat(self.imag)
        return np.identity(3) + (2*self.real) * K + 2 * K.dot(K)
        

    @staticmethod
    def exp(x):
        '''quaternion exponential, halved to match SO(3)'''

        x = np.array( x )
        theta = np.linalg.norm(x)

        res = Quaternion()
        
        if math.fabs(theta) < sys.float_info.epsilon:
            # fallback to gnomonic projection: (1 + x) / || 1 + x ||
            res.imag = x / 2.0
            res.normalize()
            return res

        half_theta = theta / 2.0
        
        s = math.sin(half_theta)
        c = math.cos(half_theta)

        res.real = c
        res.imag = x * (s / theta)

        return res

    @staticmethod
    def dexp(x):
        '''exponential derivative SO(3) in body-fixed coordinates'''

        theta = norm(x)

        if theta < Quaternion.epsilon:
            return np.identity(3)
        
        n = x / theta
        
        P = np.outer(n, n)
        H = Quaternion.hat(n)

        # we want SO(3) exponential
        theta = theta / 2.0
        
        s = math.sin(theta)
        c = math.cos(theta)

        I = np.identity(3)

        return P + (s / theta) * (c * I - s * H).dot(I - P)


    def dlog(self):
        '''logarithm derivative SO(3) in body-fixed coordinates'''
        
        n, theta = self.axis_angle()
        
        if n is None: return np.identity(3)

        theta /= 2
        res = np.zeros( (3, 3) )

        P = np.outer(n, n)

        log = n * theta
        
        return (P + (theta / math.tan(theta)) * ( np.identity(3) - P ) + Quaternion.hat(log) )

    

    def log(self):
        '''quaternion logarithm, doubled to match SO(3)'''

        axis, angle = self.axis_angle()

        if axis is None: return np.zeros(3)
        return angle * axis
    

    def axis_angle(self):
        '''rotation axis/angle'''

        q = self if self.real >= 0 else -self
        
        half_angle = math.acos( min(q.real, 1.0) )

        if half_angle > Quaternion.epsilon:
            return q.imag / math.sin(half_angle), 2 * half_angle

        n = norm(q.imag)
        if n > Quaternion.epsilon:
            sign = 1.0 if half_angle > 0 else -1.0
            return q.imag * (sign / n), 2 * half_angle
        
        return None, 2 * half_angle
    

    def angle(self):
        '''rotation angle'''
        return self.axis_angle()[1]

    def axis(self):
        '''rotation axis'''
        return self.axis_angle()[0]        

    @staticmethod
    def from_vectors(x, y):
        '''rotation sending x to y'''
        
        res = Quaternion()

        dot = x.dot(y)
        res.real = dot
        res.imag = np.cross(x, y)

        theta = norm(res)
        res.real += theta

        theta = norm(res)
        if theta < Quaternion.epsilon:

            # x == y
            if dot >= 0: return Quaternion()
            
            # x == -y
            # TODO make up vector configurable
            return Quaternion.exp( math.pi * ey )
            
        res /= theta
        return res

    
    @staticmethod
    def hat(v):
        '''cross-product matrix'''
        
        res = np.zeros( (3, 3) )

        res[:] = [[    0, -v[2],  v[1]],
                  [ v[2],     0, -v[0]],
                  [-v[1],  v[0],     0]]
        
        return res

    def slerp(self, q2, t):
        '''spherical linear interpolation between q1 and q2'''
        # TODO optimize
        return self * Quaternion.exp( t * (self.conj() * q2).log() )

    
