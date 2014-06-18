# helper functions for treating float lists as quaternions
# author: maxime.tournier@inria.fr

import sys

import math
from numpy import *
import numpy.linalg

def id():
    """identity"""
    return array([0, 0, 0, 1])

def conj(q):
    """conjugate"""
    return array([-q[0], -q[1], -q[2], q[3]])

def inv(q):
    """
    inverse 

    If you're dealing with unit quaternions, use conj instead.
    """
    return  conj(q) / numpy.linalg.norm(q)**2

def re(q):
    """real part"""
    return q[3]

def im(q):
    """imaginary part"""
    return array(q[:3])

# TODO optimize
def prod(a, b):
    """product"""
    return hstack( (re(a)*im(b) + re(b)*im(a) + numpy.cross( im(a), im(b) ), [re(a) * re(b) - dot( im(a), im(b))] ))

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
    theta = numpy.linalg.norm(v)

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
        return -1*q
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

    return (2 * half_theta / math.sin(half_theta)) * im(q)

def normalized(q):
    ## returning the normalized quaternion (without checking for a null norm...)
    return q / numpy.linalg.norm(q)

def from_matrix(M, isprecise=False):
    """Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    """

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


def to_matrix(quat):
    """Convert a quaternion into rotation matrix form.

    @param quat:    The quaternion.
    @type quat:     numpy 4D, rank-1 array
    """

    # Repetitive calculations.
    q4_2 = quat[3]**2
    q12 = quat[0] * quat[1]
    q13 = quat[0] * quat[2]
    q14 = quat[0] * quat[3]
    q23 = quat[1] * quat[2]
    q24 = quat[1] * quat[3]
    q34 = quat[2] * quat[3]

    matrix = numpy.empty((3,3))

    # The diagonal.
    matrix[0, 0] = 2.0 * (quat[0]**2 + q4_2) - 1.0
    matrix[1, 1] = 2.0 * (quat[1]**2 + q4_2) - 1.0
    matrix[2, 2] = 2.0 * (quat[2]**2 + q4_2) - 1.0

    # Off-diagonal.
    matrix[0, 1] = 2.0 * (q12 - q34)
    matrix[0, 2] = 2.0 * (q13 + q24)
    matrix[1, 2] = 2.0 * (q23 - q14)

    matrix[1, 0] = 2.0 * (q12 + q34)
    matrix[2, 0] = 2.0 * (q13 - q24)
    matrix[2, 1] = 2.0 * (q23 + q14)

    return matrix



def axisToQuat(axis, phi):
    """ return the quaternion corresponding to rotation around vector axis with angle phi
    """
    axis_norm = numpy.linalg.norm(axis)
    if axis_norm < sys.float_info.epsilon:
        return id()
    axis = axis / axis_norm
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
        axis = q[0:3]/sine
    phi =  math.acos(q[3]) * 2.0
    return [axis, phi]



def from_euler_xyz(a0,a1,a2):
    q = numpy.empty((4, ))
    q[3] = cos(a0/2.0)*cos(a1/2.0)*cos(a2/2.0) + sin(a0/2.0)*sin(a1/2.0)*sin(a2/2.0);
    q[0] = sin(a0/2.0)*cos(a1/2.0)*cos(a2/2.0) - cos(a0/2.0)*sin(a1/2.0)*sin(a2/2.0);
    q[1] = cos(a0/2.0)*sin(a1/2.0)*cos(a2/2.0) + sin(a0/2.0)*cos(a1/2.0)*sin(a2/2.0);
    q[2] = cos(a0/2.0)*cos(a1/2.0)*sin(a2/2.0) - sin(a0/2.0)*sin(a1/2.0)*cos(a2/2.0);
    return q
