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
    return im( prod(q, prod( hstack((array(x), [0])), conj(q))) )


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
        axis = numpy.asarray(q)[0:3]/sine
    phi =  math.acos(q[3]) * 2.0
    return [axis, phi]

def quatToRotVec(q):
    """ Return rotation vector corresponding to unit quaternion q
    """
    [axis, phi] = quatToAxis(q)
    return numpy.asarray(axis) * phi

def quatToRodrigues(q):
    """ Return rotation vector corresponding to unit quaternion q in the form of [axis, angle]
    """
    sine  = math.sin( math.acos(q[3]) );

    if (math.fabs(sine) < sys.float_info.epsilon) :
        axis = [0.0,1.0,0.0]
    else :
        axis = q[0:3]/sine
    phi =  math.acos(q[3]) * 2.0
    return axis * phi


#def from_euler_xyz( a ):
#    ## a is a list of 3 euler angles [x,y,z]
#    q = numpy.empty((4, ))
#    q[3] = cos(a[0]/2.0)*cos(a[1]/2.0)*cos(a[2]/2.0) + sin(a[0]/2.0)*sin(a[1]/2.0)*sin(a[2]/2.0);
#    q[0] = sin(a[0]/2.0)*cos(a[1]/2.0)*cos(a[2]/2.0) - cos(a[0]/2.0)*sin(a[1]/2.0)*sin(a[2]/2.0);
#    q[1] = cos(a[0]/2.0)*sin(a[1]/2.0)*cos(a[2]/2.0) + sin(a[0]/2.0)*cos(a[1]/2.0)*sin(a[2]/2.0);
#    q[2] = cos(a[0]/2.0)*cos(a[1]/2.0)*sin(a[2]/2.0) - sin(a[0]/2.0)*sin(a[1]/2.0)*cos(a[2]/2.0);
#    return q

#def to_euler_xyz(q):

#    norm = q[0]*q[0]+q[1]+q[1]+q[2]+q[2]

#    if math.fabs( norm ) > 1e-8 :
#        normq = norm + q[3]*q[3]
#        q /= math.sqrt(normq)
#        angle = math.acos(q[3]) * 2
#        return q[:3] / norm * angle
#    else :
#        return [0,0,0]

#    q = normalized(q)
#    angle = math.acos(q[3]) * 2;
#    v = q[:3]
#    norm = numpy.linalg.norm( v )
#    if norm > 0.0005:
#        v /= norm
#        v *= angle
#    return v


##### adapted from http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

# epsilon for testing whether a number is close to zero
_EPS = numpy.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())



def euler_from_matrix(M, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print(axes, "failed")

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

#    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    a = numpy.empty((3, ))

    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            a[0] = math.atan2( M[i, j],  M[i, k])
            a[1] = math.atan2( sy,       M[i, i])
            a[2] = math.atan2( M[j, i], -M[k, i])
        else:
            a[0] = math.atan2(-M[j, k],  M[j, j])
            a[1] = math.atan2( sy,       M[i, i])
            a[2] = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
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

def to_euler(q, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.
    """
    return euler_from_matrix( to_matrix(q), axes )


def from_euler( a, axes='sxyz' ):
    """Return quaternion from Euler angles and axis sequence.

    a is a list of 3 euler angles [x,y,z]
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

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

    q = numpy.empty((4, ))
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
