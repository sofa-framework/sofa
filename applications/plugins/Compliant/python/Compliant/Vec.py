# basic euclidean space operations on float lists.
# 
# author: maxime.tournier@inria.fr


import math

import __builtin__ as base

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

def sum(x, y):
    return [xi + yi for xi, yi in zip(x, y) ]

def diff(x, y):
    return [xi - yi for xi, yi in zip(x, y) ]

# TODO assert dimension is 3
def cross(x, y):
    return [ x[1] * y[2] - x[2] * y[1],
             x[2] * y[0] - x[0] * y[2],
             x[0] * y[1] - x[1] * y[0] ]

# eases long operations
class Proxy:
    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        return Proxy( sum(self.data, other.data) )

    def __sub__(self, other):
        return Proxy( diff(self.data, other.data) )

    def __rmul__(self, scalar):
        return Proxy( scal(scalar, self.data) )

    def norm2(self): 
        return norm2(self.data)

    def norm(self): 
        return norm(self.data)

    def dot(self, other): 
        return dot(self.data, other.data)


    


