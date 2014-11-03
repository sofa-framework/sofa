# helper functions for treating float lists as quaternions
# author: maxime.tournier@inria.fr

import sys

import Vec
import math
import numpy

print "WARNING Compliant's Quaternion.py is now deprecated (and will be deleted soon), please use SofaPython's one instead"


def id():
    """identity"""
    return [0, 0, 0, 1]

def conj(q):
    """conjugate"""
    return [-q[0], -q[1], -q[2], q[3]]

def inv(q):
    """
    inverse 

    If you're dealing with unit quaternions, use conj instead.
    """
    return Vec.scal(1 / Vec.norm2(q), conj(q) )

def re(q):
    """real part"""
    return q[3]

def im(q):
    """imaginary part"""
    return q[:3]

# TODO optimize
def prod(a, b):
    """product"""
    return Vec.sum( Vec.sum( Vec.scal( re(a), im(b) ),
                             Vec.scal( re(b), im(a) )),
                    Vec.cross( im(a), im(b) ) ) + [ re(a) * re(b) - Vec.dot( im(a), im(b) ) ]

# TODO optimize
def rotate(q, x):
    """vector rotation

    rotates x by the rotation represented by q. this is also the
    adjoint map for S^3.

    """
    
    # TODO assert q is unit
    return im( prod(q, prod( x + [0], conj(q))) )


def exp(v):
    """exponential"""
    theta = Vec.norm(v)

    if math.fabs(theta) < sys.float_info.epsilon:
        return id()

    s = math.sin(theta / 2)
    c = math.cos(theta / 2)

    return [ v[0] / theta * s,
             v[1] / theta * s,
             v[2] / theta * s,
             c ]

def flip(q):
    """Flip a quaternion to the real positive hemisphere if needed."""
    
    if re(q) < 0:
        return Vec.minus(q)
    else :
        return q

    
def log(q):

    """(principal) logarithm. 

    You might want to flip q first to ensure theta is in the [-0, pi]
    range, yielding the equivalent rotation (principal) logarithm.

    """

    half_theta = math.acos( sign * re(q) )

    if math.fabs( half_theta ) < sys.float_info.epsilon:
        return [0, 0, 0]

    return Vec.scal(2 * half_theta / math.sin(half_theta),
                    im(q))



def from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]

    if q[0] < 0.0:
        numpy.negative(q, q)

    #return q.tolist()
    return [ q[1], q[2], q[3], q[0] ] # sofa order

def axisToQuat(axis, phi):
    """ return the quaternion corresponding to rotation around vector axis with angle phi
    """
    axis_norm = Vec.norm(axis)
    if axis_norm < sys.float_info.epsilon:
        return id()
    axis = Vec.scal(1./axis_norm, axis)
    return [ axis[0]*math.sin(phi/2),
             axis[1]*math.sin(phi/2),
             axis[2]*math.sin(phi/2),
             math.cos(phi/2) ]

def quatToAxis(q):
    """ Return rotation vector corresponding to unit quaternion q in the form of [axis, angle]
    """
    sine  = math.sin( math.acos(q[3]) );

    if (math.fabs(sine) < sys.float_info.epsilon) :
        axis = [0.0,1.0,0.0]
    else :
        axis = Vec.scal(1/sine, q[0:3])
    phi =  math.acos(q[3]) * 2.0
    return [axis, phi]
