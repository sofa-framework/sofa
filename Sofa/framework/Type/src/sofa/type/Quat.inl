/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/type/Quat.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>

#include <limits>
#include <cmath>
#include <iostream>
#include <cstdio>

namespace sofa::type
{

static constexpr SReal quaternionEqualityThreshold=1e-6;

template<class Real>
Quat<Real>::Quat( const Vec3& axis, Real angle )
{
    axisToQuat(axis,angle);
}

template<class Real>
Quat<Real>::Quat(const Vec3& vFrom, const Vec3& vTo)
{
    setFromUnitVectors(vFrom, vTo);
}

/// Given two rotations, e1 and e2, expressed as quaternion rotations,
/// figure out the equivalent single rotation and stuff it into dest.
///   NOTE: This routine is written so that q1 or q2 may be the same
///  	   as dest (or each other).
template<class Real>
auto Quat<Real>::operator+(const Quat &q1) const -> Quat
{
    Real		t1[4], t2[4], t3[4];
    Real		tf[4];
    Quat    	ret(QNOINIT);

    t1[0] = _q[0] * q1._q[3];
    t1[1] = _q[1] * q1._q[3];
    t1[2] = _q[2] * q1._q[3];

    t2[0] = q1._q[0] * _q[3];
    t2[1] = q1._q[1] * _q[3];
    t2[2] = q1._q[2] * _q[3];

    // cross product t3 = q2 x q1
    t3[0] = (q1._q[1] * _q[2]) - (q1._q[2] * _q[1]);
    t3[1] = (q1._q[2] * _q[0]) - (q1._q[0] * _q[2]);
    t3[2] = (q1._q[0] * _q[1]) - (q1._q[1] * _q[0]);
    // end cross product

    tf[0] = t1[0] + t2[0] + t3[0];
    tf[1] = t1[1] + t2[1] + t3[1];
    tf[2] = t1[2] + t2[2] + t3[2];
    tf[3] = _q[3] * q1._q[3] -
            (_q[0] * q1._q[0] + _q[1] * q1._q[1] + _q[2] * q1._q[2]);

    ret._q[0] = tf[0];
    ret._q[1] = tf[1];
    ret._q[2] = tf[2];
    ret._q[3] = tf[3];

    ret.normalize();

    return ret;
}

template<class Real>
auto Quat<Real>::inverse() const -> Quat
{
    Quat	ret(QNOINIT);
    Real		norm	= sqrt(_q[0] * _q[0] +
            _q[1] * _q[1] +
            _q[2] * _q[2] +
            _q[3] * _q[3]);

    if (norm != 0.0f)
    {
        norm = 1.0f / norm;
        ret._q[3] = _q[3] * norm;
        for (int i = 0; i < 3; i++)
        {
            ret._q[i] = -_q[i] * norm;
        }
    }
    else
    {
        for (int i = 0; i < 4; i++)
        {
            ret._q[i] = 0.0;
        }
    }

    return ret;
}

/// Returns true if norm of Quatnion is one, false otherwise.
template<class Real>
bool Quat<Real>::isNormalized()
{
    Real mag = (_q[0] * _q[0] + _q[1] * _q[1] + _q[2] * _q[2] + _q[3] * _q[3]);
    Real epsilon = std::numeric_limits<Real>::epsilon();
    return (std::abs(mag - 1.0) < epsilon);
}


/// Quat<Real>s always obey:  a^2 + b^2 + c^2 + d^2 = 1.0
/// If they don't add up to 1.0, dividing by their magnitude will
/// renormalize them.
template<class Real>
void Quat<Real>::normalize()
{
    const Real mag = (_q[0] * _q[0] + _q[1] * _q[1] + _q[2] * _q[2] + _q[3] * _q[3]);
    Real epsilon = static_cast<Real>(1.0e-10);
    if (std::abs(mag - 1.0) > epsilon)
    {
        if( mag != 0)
        {
            const Real sqr = static_cast<Real>(1.0 / sqrt(mag));
            for (auto& i : _q)
            {
                i *= sqr;
            }
        }
        else
        {
            _q[3] = 1;
        }
    }
}

template<class Real>
void Quat<Real>::fromFrame(const Vec3& x, const Vec3&y, const Vec3&z)
{
    Mat3x3 R(x,y,z);
    R.transpose();
    this->fromMatrix(R);
}

template<class Real>
void Quat<Real>::fromMatrix(const Mat3x3 &m)
{
    Real tr, s;
    tr = m.x().x() + m.y().y() + m.z().z();

    // check the diagonal
    if (tr > 0)
    {
        s = sqrt (tr + 1);
        _q[3] = s * 0.5f; // w OK
        s = 0.5f / s;
        _q[0] = (m.z().y() - m.y().z()) * s; // x OK
        _q[1] = (m.x().z() - m.z().x()) * s; // y OK
        _q[2] = (m.y().x() - m.x().y()) * s; // z OK
    }
    else
    {
        if (m.y().y() > m.x().x() && m.z().z() <= m.y().y())
        {
            s = sqrt ((m.y().y() - (m.z().z() + m.x().x())) + 1.0f);

            _q[1] = s * 0.5f; // y OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[2] = (m.y().z() + m.z().y()) * s; // z OK
            _q[0] = (m.x().y() + m.y().x()) * s; // x OK
            _q[3] = (m.x().z() - m.z().x()) * s; // w OK
        }
        else if ((m.y().y() <= m.x().x()  &&  m.z().z() > m.x().x())  ||  (m.z().z() > m.y().y()))
        {
            s = sqrt ((m.z().z() - (m.x().x() + m.y().y())) + 1.0f);

            _q[2] = s * 0.5f; // z OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[0] = (m.z().x() + m.x().z()) * s; // x OK
            _q[1] = (m.y().z() + m.z().y()) * s; // y OK
            _q[3] = (m.y().x() - m.x().y()) * s; // w OK
        }
        else
        {
            s = sqrt ((m.x().x() - (m.y().y() + m.z().z())) + 1.0f);

            _q[0] = s * 0.5f; // x OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[1] = (m.x().y() + m.y().x()) * s; // y OK
            _q[2] = (m.z().x() + m.x().z()) * s; // z OK
            _q[3] = (m.z().y() - m.y().z()) * s; // w OK
        }
    }
}

/// Given an axis and angle, compute quaternion.
template<class Real>
auto Quat<Real>::axisToQuat(Vec3 a, Real phi) -> Quat
{
    const auto aNorm = a.norm();
    if(aNorm < std::numeric_limits<Real>::epsilon() )
    {
        _q[0] = _q[1] = _q[2] = Real(0.0);
        _q[3] = Real(1.0);

        return Quat();
    }

    a = a / aNorm;
    const auto sp = sin(phi / Real(2.0));
    const auto cp = cos(phi / Real(2.0));

    _q[0] = a.x() * sp;
    _q[1] = a.y() * sp;
    _q[2] = a.z() * sp;
    _q[3] = cp;

    return *this;
}

/// Given a quaternion, compute an axis and angle
template<class Real>
void Quat<Real>::quatToAxis(Vec3 & axis, Real &angle) const
{
    Quat<Real> q = *this;
    if(q[3]<0)
        q*=-1; // we only work with theta in [0, PI]

    Real sin_half_theta; // note that sin(theta/2) == norm of the imaginary part for unit quaternion

    // to avoid numerical instabilities of acos for theta < 5°
    if(q[3]>0.999) // theta < 5° -> q[3] = cos(theta/2) > 0.999
    {
        sin_half_theta = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
        angle = static_cast<Real>(2.0 * asin(sin_half_theta));
    }
    else
    {
        Real half_theta = acos(q[3]);
        sin_half_theta = sin(half_theta);
        angle = 2*half_theta;
    }

    assert(sin_half_theta>=0);
    if (sin_half_theta < std::numeric_limits<Real>::epsilon())
        axis = Vec3(Real(0.), Real(1.), Real(0.));
    else
        axis = Vec3(q[0], q[1], q[2])/sin_half_theta;
}

/// Given a quaternion, compute rotation vector (axis times angle)
template<class Real>
auto Quat<Real>::quatToRotationVector() const -> Vec3
{
    Quat q = *this;
    q.normalize();

    Real angle;

    if(q[3]<0)
        q*=-1; // we only work with theta in [0, PI] (i.e. angle in [0, 2*PI])

    Real sin_half_theta; // note that sin(theta/2) == norm of the imaginary part for unit quaternion

    // to avoid numerical instabilities of acos for theta < 5°
    if(q[3]>0.999) // theta < 5° -> q[3] = cos(theta/2) > 0.999
    {
        sin_half_theta = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
        angle = static_cast<Real>(2.0 * asin(sin_half_theta));
    }
    else
    {
        Real half_theta = acos(q[3]);
        sin_half_theta = sin(half_theta);
        angle = 2*half_theta;
    }

    assert(sin_half_theta>=0);
    Vec3 rotVector;
    if (sin_half_theta < std::numeric_limits<Real>::epsilon())
        rotVector = Vec3(Real(0), Real(0), Real(0));
    else
        rotVector = Vec3(q[0], q[1], q[2])/sin_half_theta*angle;

    return rotVector;
}


/// Compute the Euler angles:
/// Roll: rotation about the X-axis
/// Pitch: rotation about the Y-axis
/// Yaw: rotation about the Z-axis
template<class Real>
auto Quat<Real>::toEulerVector() const -> Vec3
{
    Quat q = *this;
    q.normalize();

    // Cancel numerical drifting by clamping on [-1 ; 1]
    Real y = std::max(Real(-1.0), std::min(Real(1.0), Real(2.)*(q[3]*q[1] - q[2]*q[0])));

    Vec3 vEuler;
    vEuler[0] = atan2(2*(q[3]*q[0] + q[1]*q[2]) , (1-2*(q[0]*q[0] + q[1]*q[1])));   //roll
    vEuler[1] = asin(y); // pitch
    vEuler[2] = atan2(2*(q[3]*q[2] + q[0]*q[1]) , (1-2*(q[1]*q[1] + q[2]*q[2])));   //yaw
    return vEuler;
}

/*! Returns the slerp interpolation of Quaternions \p a and \p b, at time \p t.

 \p t should range in [0,1]. Result is \p a when \p t=0 and \p b when \p t=1.

 When \p allowFlip is \c true (default) the slerp interpolation will always use the "shortest path"
 between the Quaternions' orientations, by "flipping" the source Quatnion if needed (see
 negate()). */
template<class Real>
void Quat<Real>::slerp(const Quat& a, const Quat& b, Real t, bool allowFlip)
{
    Real cosAngle =  static_cast<Real>(a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]);

    Real c1, c2;
    // Linear interpolation for close orientations
    if ((1.0 - std::abs(cosAngle)) < 0.01)
    {
        c1 = 1.0f - t;
        c2 = t;
    }
    else
    {
        // Spherical interpolation
        Real angle    = acos(std::abs(cosAngle));
        Real sinAngle = sin(angle);
        c1 = sin(angle * (1.0f - t)) / sinAngle;
        c2 = sin(angle * t) / sinAngle;
    }

    // Use the shortest path
    if (allowFlip && (cosAngle < 0.0f))
        c1 = -c1;

    _q[0] = c1*a[0] + c2*b[0];
    _q[1] = c1*a[1] + c2*b[1];
    _q[2] = c1*a[2] + c2*b[2];
    _q[3] = c1*a[3] + c2*b[3];
}

template<class Real>
auto Quat<Real>::slerp(const Quat &q1, Real t) const -> Quat
{
    Quat q0_1(QNOINIT);
    for (unsigned int i = 0 ; i<3 ; i++)
        q0_1[i] = -_q[i];

    q0_1[3] = _q[3];

    q0_1 = q1 * q0_1;

    Vec3 axis(NOINIT), temp(NOINIT);
    Real angle;

    q0_1.quatToAxis(axis, angle);

    temp = axis * sin(t * angle);
    for (unsigned int i = 0 ; i<3 ; i++)
        q0_1[i] = temp[i];

    q0_1[3] = cos(t * angle);
    q0_1 = q0_1 * (*this);
    return q0_1;
}

// Given an axis and angle, compute quaternion.
template<class Real>
auto Quat<Real>::slerp2(const Quat &q1, Real t) const -> Quat
{
    // quaternion to return
    Quat qm(QNOINIT);

    // Calculate angle between them.
    Real cosHalfTheta = _q[3] * q1[3] + _q[0] * q1[0] + _q[1] * q1[1] + _q[2] * q1[2];
    // if qa=qb or qa=-qb then theta = 0 and we can return qa
    if (std::abs(cosHalfTheta) >= 1.0)
    {
        qm[3] = _q[3]; qm[0] = _q[0]; qm[1] = _q[1]; qm[2] = _q[2];
        return qm;
    }
    // Calculate temporary values.
    Real halfTheta = acos(cosHalfTheta);
    Real sinHalfTheta = static_cast<Real>(sqrt(1.0 - cosHalfTheta*cosHalfTheta));
    // if theta = 180 degrees then result is not fully defined
    // we could rotate around any axis normal to qa or qb
    if (std::abs(sinHalfTheta) < 0.001)
    {
        qm[3] = static_cast<Real>(_q[3] * 0.5 + q1[3] * 0.5);
        qm[0] = static_cast<Real>(_q[0] * 0.5 + q1[0] * 0.5);
        qm[1] = static_cast<Real>(_q[1] * 0.5 + q1[1] * 0.5);
        qm[2] = static_cast<Real>(_q[2] * 0.5 + q1[2] * 0.5);
        return qm;
    }
    Real ratioA = sin((1 - t) * halfTheta) / sinHalfTheta;
    Real ratioB = sin(t * halfTheta) / sinHalfTheta;
    //calculate Quatnion.
    qm[3] = static_cast<Real>(_q[3] * ratioA + q1[3] * ratioB);
    qm[0] = static_cast<Real>(_q[0] * ratioA + q1[0] * ratioB);
    qm[1] = static_cast<Real>(_q[1] * ratioA + q1[1] * ratioB);
    qm[2] = static_cast<Real>(_q[2] * ratioA + q1[2] * ratioB);
    return qm;

}

template<class Real>
auto Quat<Real>::createQuaterFromFrame(const Vec3 &lox, const Vec3 &loy,const Vec3 &loz) -> Quat
{
    Quat q(QNOINIT);
    Mat3x3 m;

    for (unsigned int i=0 ; i<3 ; i++)
    {
        m(i,0) = lox[i];
        m(i,1) = loy[i];
        m(i,2) = loz[i];
    }
    q.fromMatrix(m);
    return q;
}

template<class Real>
void Quat<Real>::setFromUnitVectors(const Vec3& vFrom, const Vec3& vTo)
{
    Vec3 v1;
    Real epsilon = Real(0.0001);
    
    Real res_dot = type::dot(vFrom, vTo) + 1;
    if (res_dot < epsilon)
    {
        res_dot = 0;
        if (fabs(vFrom[0]) > fabs(vFrom[2]))
            v1 = Vec3(-vFrom[1], vFrom[0], Real(0.));
        else
            v1 = Vec3(Real(0.), -vFrom[2], vFrom[1]);
    }
    else
    {
        v1 = vFrom.cross(vTo);
    }

    _q[0] = v1[0];
    _q[1] = v1[1];
    _q[2] = v1[2];
    _q[3] = res_dot;

    this->normalize();
}

template<class Real>
void Quat<Real>::operator+=(const Quat& q2)
{
    Real t1[4], t2[4], t3[4];
    Quat q1 = (*this);
    t1[0] = q1._q[0] * q2._q[3];
    t1[1] = q1._q[1] * q2._q[3];
    t1[2] = q1._q[2] * q2._q[3];

    t2[0] = q2._q[0] * q1._q[3];
    t2[1] = q2._q[1] * q1._q[3];
    t2[2] = q2._q[2] * q1._q[3];

    // cross product t3 = q2 x q1
    t3[0] = (q2._q[1] * q1._q[2]) - (q2._q[2] * q1._q[1]);
    t3[1] = (q2._q[2] * q1._q[0]) - (q2._q[0] * q1._q[2]);
    t3[2] = (q2._q[0] * q1._q[1]) - (q2._q[1] * q1._q[0]);
    // end cross product

    _q[0] = t1[0] + t2[0] + t3[0];
    _q[1] = t1[1] + t2[1] + t3[1];
    _q[2] = t1[2] + t2[2] + t3[2];
    _q[3] = q1._q[3] * q2._q[3] -
            (q1._q[0] * q2._q[0] + q1._q[1] * q2._q[1] + q1._q[2] * q2._q[2]);

    normalize();
}

template<class Real>
Quat<Real> Quat<Real>::quatDiff(Quat a, const Quat& b)
{
    // If the axes are not oriented in the same direction, flip the axis and angle of a to get the same convention than b
    if (a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3]<0)
    {
        a[0] = -a[0];
        a[1] = -a[1];
        a[2] = -a[2];
        a[3] = -a[3];
    }

    Quat q = b.inverse() * a;
    return q;
}

/// Return the eulerian vector resulting of the movement between 2 quaternions
template<class Real>
auto Quat<Real>::angularDisplacement(const Quat& a, const Quat& b) -> Vec3
{
    // In the following, use of quatToRotationVector instead of toEulerVector:
    // this is done to keep the old behavior (before the correction of the toEulerVector function).
    return quatDiff(a,b).quatToRotationVector();
}

template<class Real>
auto Quat<Real>::createFromRotationVector(Real a0, Real a1, Real a2 ) -> Quat
{
    Real phi = Real(sqrt(a0*a0+a1*a1+a2*a2));

    if( phi >= 1.0e-5 )
        return Quat(0,0,0,1);

    Real nor = 1/phi;
    Real s = sin(phi/Real(2.));
    return Quat( a0*s*nor, a1*s*nor,a2*s*nor, cos(phi/Real(2.)) );
}

template<class Real>
auto Quat<Real>::fromEuler( Real alpha, Real beta, Real gamma, EulerOrder order) -> Quat
{
    return createQuaterFromEuler( {alpha, beta, gamma }, order );
}

template<class Real>
auto Quat<Real>::createFromRotationVector(const Vec3& a) -> Quat
{
    Real phi = Real(sqrt(a*a));
    if( phi <= 1.0e-5 )
        return Quat(0,0,0,1);

    Real nor = 1/phi;
    Real s = Real(sin(phi/2));
    return Quat( a[0]*s*nor, a[1]*s*nor,a[2]*s*nor, Real(cos(phi/2)));
}


template<class Real>
auto Quat<Real>::createQuaterFromEuler(const Vec3& v, EulerOrder order) -> Quat
{
    Real quat[4];

    Real c1 = cos( v.elems[0] / 2 );
    Real c2 = cos( v.elems[1] / 2 );
    Real c3 = cos( v.elems[2] / 2 );

    Real s1 = sin( v.elems[0] / 2 );
    Real s2 = sin( v.elems[1] / 2 );
    Real s3 = sin( v.elems[2] / 2 );

    switch(order)
    {
    case EulerOrder::XYZ:
        quat[0] = s1 * c2 * c3 + c1 * s2 * s3;
        quat[1] = c1 * s2 * c3 - s1 * c2 * s3;
        quat[2] = c1 * c2 * s3 + s1 * s2 * c3;
        quat[3] = c1 * c2 * c3 - s1 * s2 * s3;
        break;
    case EulerOrder::YXZ:
        quat[0] = s1 * c2 * c3 + c1 * s2 * s3;
        quat[1] = c1 * s2 * c3 - s1 * c2 * s3;
        quat[2] = c1 * c2 * s3 - s1 * s2 * c3;
        quat[3] = c1 * c2 * c3 + s1 * s2 * s3;
        break;
    case EulerOrder::ZXY:
        quat[0] = s1 * c2 * c3 - c1 * s2 * s3;
        quat[1] = c1 * s2 * c3 + s1 * c2 * s3;
        quat[2] = c1 * c2 * s3 + s1 * s2 * c3;
        quat[3] = c1 * c2 * c3 - s1 * s2 * s3;
        break;
    case EulerOrder::YZX:
        quat[0] = s1 * c2 * c3 + c1 * s2 * s3;
        quat[1] = c1 * s2 * c3 + s1 * c2 * s3;
        quat[2] = c1 * c2 * s3 - s1 * s2 * c3;
        quat[3] = c1 * c2 * c3 - s1 * s2 * s3;
        break;
    case EulerOrder::XZY:
        quat[0] = s1 * c2 * c3 - c1 * s2 * s3;
        quat[1] = c1 * s2 * c3 - s1 * c2 * s3;
        quat[2] = c1 * c2 * s3 + s1 * s2 * c3;
        quat[3] = c1 * c2 * c3 + s1 * s2 * s3;
        break;
    default:
    case EulerOrder::ZYX:
        quat[0] = s1 * c2 * c3 - c1 * s2 * s3;
        quat[1] = c1 * s2 * c3 + s1 * c2 * s3;
        quat[2] = c1 * c2 * s3 - s1 * s2 * c3;
        quat[3] = c1 * c2 * c3 + s1 * s2 * s3;
        break;
    }

    Quat quatResult{ quat[0], quat[1], quat[2], quat[3] };
    return quatResult;
}

template<class Real>
bool Quat<Real>::operator==(const Quat& q) const
{
    for (int i=0; i<4; i++)
        if ( std::abs( _q[i] - q._q[i] ) > quaternionEqualityThreshold) return false;
    return true;
}

template<class Real>
bool Quat<Real>::operator!=(const Quat& q) const
{
    for (int i=0; i<4; i++)
        if ( std::abs( _q[i] - q._q[i] ) > quaternionEqualityThreshold) return true;
    return false;
}

/// write to an output stream
template<class Real>
std::ostream& operator << (std::ostream& out, const Quat<Real>& v)
{
    out << v[0] << " " << v[1] << " " << v[2] << " " << v[3];
    return out;
}

/// read from an input stream
template<class Real>
std::istream& operator >> (std::istream& in, Quat<Real>& v)
{
    in >> v[0] >> v[1] >> v[2] >> v[3];
    return in;
}

} // namespace sofa::type
