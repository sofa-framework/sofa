# basic euclidean space operations on float lists.
# 
# author: maxime.tournier@inria.fr


import math

import __builtin__ as base

print "WARNING Compliant's Vec.py is now deprecated (and will be deleted soon), please use numpy instead"

def dot(x, y):
    return base.sum( [xi * yi for xi, yi in zip(x, y) ] )

def norm2(x):
    return base.sum( [ xi * xi for xi in x] )

def norm(x):
    return math.sqrt( norm2(x) )

def zero(n):
    return [0.0] * n

def scal( scalar, x ):
    return [ scalar * xi for xi in x ]

def minus(x):
    return scal(-1, x)

def inv(x):
    return [1.0/xi for xi in x ]

def sum(x, y):
    return [xi + yi for xi, yi in zip(x, y) ]

def diff(x, y):
    return [xi - yi for xi, yi in zip(x, y) ]

# TODO assert dimension is 3
def cross(x, y):
    return [ x[1] * y[2] - x[2] * y[1],
             x[2] * y[0] - x[0] * y[2],
             x[0] * y[1] - x[1] * y[0] ]

# eases vector operations. you might want to use it as follows:
#
# from Vec import Proxy as vec
# 
# v = vec( [2, 4, 2] )
#
 
class Proxy:
    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        return Proxy( sum(self.data, other.data) )

    def __sub__(self, other):
        return Proxy( diff(self.data, other.data) )

    def __rmul__(self, scalar):
        return Proxy( scal(scalar, self.data) )

    def __div__(self, scalar):
        return Proxy( scal(1 / scalar, self.data) )

    def __neg__(self):
        return Proxy( minus(self.data) )

    def norm2(self): 
        return norm2(self.data)

    def norm(self): 
        return norm(self.data)

    def dot(self, other): 
        return dot(self.data, other.data)

    def cross(self, other):
        return Proxy(cross(self.data, other.data))

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __str__(self):
        return ' '.join(map(str, self.data))

    def copy(self):
        return Proxy( [xi for xi in self.data] )
