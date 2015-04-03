import numpy as np
import math
import sys

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
        res.center = -res.orient.rotate(self.center)
        return res

    def __mul__(self, other):
        res = Rigid3()

        res.orient = self.orient * other.orient
        res.center = self.center + self.orient.rotate(other.center)
        
        return res


    def __call__(self, x):
        '''applies rigid transform to vector x'''
        return self.center + self.orient(x)
    

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
    
    @staticmethod
    def exp(x):
        '''quaternion exponential (doubled)'''

        theta = np.linalg.norm(x)

        res = Quaternion()
        
        if math.fabs(theta) < sys.float_info.epsilon:
            return res

        s = math.sin(theta / 2.0)
        c = math.cos(theta / 2.0)

        res.real = c
        res.imag = x * (s / theta)

        return res
