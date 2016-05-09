import numpy as np
import math
import sys

def vec(*coords):
    return np.array(coords)

class Rigid3(np.ndarray):

    @property
    def center(self):
        return self[:3]

    @center.setter
    def center(self, value):
        self[:3] = value

    @property
    def orient(self):
        return self[3:].view(Quaternion)

    @orient.setter
    def orient(self, value):
        self[3:] = value

    def __new__(cls, *args):
        return np.ndarray.__new__(cls, 7)
        
    def __init__(self):
        self[-1] = 1
        self[:6] = 0

    def inv(self):
        res = Rigid3()
        res.orient = self.orient.inv()
        res.center = -res.orient(self.center)
        return res

    def __mul__(self, other):
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

        res[:3, :3] = R
        res[3:, 3:] = R

        res[3:, :3] = t.dot(R)

        return res

    def matrix(self):
        '''homogeneous matrix for rigid transformation'''

        res = np.zeros( (4, 4) )

        res[:3, :3] = self.orient.matrix()
        res[:3, 3] = self.center

        res[3, 3] = 1

        return res
        
class Quaternion(np.ndarray):
    
    def __new__(cls, *args):
        return np.ndarray.__new__(cls, 4)
        
    def __init__(self):
        '''identity quaternion'''
        self.real = 1
        self.imag = 0
        
    def inv(self):
        '''inverse'''
        return self.conj() / self.dot(self)
    
    def conj(self):
        '''conjugate'''
        res = Quaternion()
        res.real = self.real
        res.imag = -self.imag

        return res

    @property
    def real(self): return self[-1]

    @real.setter
    def real(self, value): self[-1] = value

    @property
    def imag(self): return self[:3]

    @imag.setter
    def imag(self, value): self[:3] = value

    def normalize(self):
        '''normalize quaternion'''
        self /= math.sqrt( self.dot(self) )

    def flip(self):
        '''flip quaternion in the real positive halfplane, if needed'''
        if self.real < 0: self = -self

    def __mul__(self, other):
        '''quaternion product'''
        res = Quaternion()
        res.real = self.real * other.real - self.imag.dot(other.imag)
        res.imag = self.real * other.imag + other.real * self.imag + np.cross(self.imag, other.imag)
        
        return res
         

    def __call__(self, x):
        '''rotate a vector. self should be normalized'''
        
        tmp = Quaternion()
        tmp.real = 0
        tmp.imag = x

        return (self * tmp * self.conj()).imag


    # TODO this is horribly inefficient, optimize
    def matrix(self):
        '''rotation matrix'''

        R = np.identity(3)

        for i in range(3):
            R[:, i] = self( np.eye(1, 3, i) )

        return R
        
    
    @staticmethod
    def exp(x):
        '''quaternion exponential (doubled)'''

        x = np.array( x )
        theta = np.linalg.norm(x)

        res = Quaternion()
        
        if math.fabs(theta) < sys.float_info.epsilon:
            return res

        s = math.sin(theta / 2.0)
        c = math.cos(theta / 2.0)

        res.real = c
        res.imag = x * (s / theta)

        return res


    @staticmethod
    def hat(v):
        '''cross-product matrix'''
        
        res = np.zeros( (3, 3) )

        res[0, 1] = -v[2]
        res[0, 2] = v[1]
        res[1, 2] = -v[0]

        res -= res.T

        return res


    
