/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_QUATER_INL
#define SOFA_HELPER_QUATER_INL

#include "Quater.h"
#include <limits>
#include <cmath>
#include <iostream>
#include <cstdio>


namespace sofa
{

namespace helper
{

#define RENORMCOUNT 50

// Constructor
template<class Real>
Quater<Real>::Quater()
{
    clear();
}

template<class Real>
Quater<Real>::Quater(Real x, Real y, Real z, Real w)
{
    set(x,y,z,w);
}

template<class Real>
Quater<Real>::Quater( const defaulttype::Vec<3,Real>& axis, Real angle )
{
    axisToQuat(axis,angle);
}

// Destructor
template<class Real>
Quater<Real>::~Quater()
{
}

/// Given two rotations, e1 and e2, expressed as quaternion rotations,
/// figure out the equivalent single rotation and stuff it into dest.
/// This routine also normalizes the result every RENORMCOUNT times it is
/// called, to keep error from creeping in.
///   NOTE: This routine is written so that q1 or q2 may be the same
///  	   as dest (or each other).
template<class Real>
//Quater<Real> operator+(Quater<Real> q1, Quater<Real> q2) const
Quater<Real> Quater<Real>::operator+(const Quater<Real> &q1) const
{
//    static int	count	= 0;

    Real		t1[4], t2[4], t3[4];
    Real		tf[4];
    Quater<Real>	ret;

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

/*    if (++count > RENORMCOUNT)
    {
        count = 0;
        ret.normalize();
    } */

	ret.normalize();

    return ret;
}

template<class Real>
//Quater<Real> operator*(const Quater<Real>& q1, const Quater<Real>& q2) const
Quater<Real> Quater<Real>::operator*(const Quater<Real>& q1) const
{
    Quater<Real>	ret;

    ret._q[3] = _q[3] * q1._q[3] -
            (_q[0] * q1._q[0] +
                    _q[1] * q1._q[1] +
                    _q[2] * q1._q[2]);
    ret._q[0] = _q[3] * q1._q[0] +
            _q[0] * q1._q[3] +
            _q[1] * q1._q[2] -
            _q[2] * q1._q[1];
    ret._q[1] = _q[3] * q1._q[1] +
            _q[1] * q1._q[3] +
            _q[2] * q1._q[0] -
            _q[0] * q1._q[2];
    ret._q[2] = _q[3] * q1._q[2] +
            _q[2] * q1._q[3] +
            _q[0] * q1._q[1] -
            _q[1] * q1._q[0];

    return ret;
}

template<class Real>
Quater<Real> Quater<Real>::operator*(const Real& r) const
{
    Quater<Real>  ret;
    ret[0] = _q[0] * r;
    ret[1] = _q[1] * r;
    ret[2] = _q[2] * r;
    ret[3] = _q[3] * r;
    return ret;
}


template<class Real>
Quater<Real> Quater<Real>::operator/(const Real& r) const
{
    Quater<Real>  ret;
    ret[0] = _q[0] / r;
    ret[1] = _q[1] / r;
    ret[2] = _q[2] / r;
    ret[3] = _q[3] / r;
    return ret;
}

template<class Real>
void Quater<Real>::operator*=(const Real& r)
{
    Quater<Real>  ret;
    _q[0] *= r;
    _q[1] *= r;
    _q[2] *= r;
    _q[3] *= r;
}


template<class Real>
void Quater<Real>::operator/=(const Real& r)
{
    Quater<Real>  ret;
    _q[0] /= r;
    _q[1] /= r;
    _q[2] /= r;
    _q[3] /= r;
}


template<class Real>
Quater<Real> Quater<Real>::quatVectMult(const defaulttype::Vec<3,Real>& vect)
{
    Quater<Real>	ret;

    ret._q[3] = (Real) (-(_q[0] * vect[0] + _q[1] * vect[1] + _q[2] * vect[2]));
    ret._q[0] = (Real) (_q[3] * vect[0] + _q[1] * vect[2] - _q[2] * vect[1]);
    ret._q[1] = (Real) (_q[3] * vect[1] + _q[2] * vect[0] - _q[0] * vect[2]);
    ret._q[2] = (Real) (_q[3] * vect[2] + _q[0] * vect[1] - _q[1] * vect[0]);

    return ret;
}

template<class Real>
Quater<Real> Quater<Real>::vectQuatMult(const defaulttype::Vec<3,Real>& vect)
{
    Quater<Real>	ret;

    ret[3] = (Real) (-(vect[0] * _q[0] + vect[1] * _q[1] + vect[2] * _q[2]));
    ret[0] = (Real) (vect[0] * _q[3] + vect[1] * _q[2] - vect[2] * _q[1]);
    ret[1] = (Real) (vect[1] * _q[3] + vect[2] * _q[0] - vect[0] * _q[2]);
    ret[2] = (Real) (vect[2] * _q[3] + vect[0] * _q[1] - vect[1] * _q[0]);

    return ret;
}

template<class Real>
Quater<Real> Quater<Real>::inverse() const
{
    Quater<Real>	ret;

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

/// Quater<Real>s always obey:  a^2 + b^2 + c^2 + d^2 = 1.0
/// If they don't add up to 1.0, dividing by their magnitude will
/// renormalize them.
template<class Real>
void Quater<Real>::normalize()
{
    Real mag = (_q[0] * _q[0] + _q[1] * _q[1] + _q[2] * _q[2] + _q[3] * _q[3]);
    if( mag != 0)
    {
        Real sqr = static_cast<Real>(1.0 / sqrt(mag));
        for (int i = 0; i < 4; i++)
        {
            _q[i] *= sqr;
        }
    }
}

template<class Real>
void Quater<Real>::fromFrame(defaulttype::Vec<3,Real>& x, defaulttype::Vec<3,Real>&y, defaulttype::Vec<3,Real>&z)
{

    defaulttype::Matrix3 R(x,y,z);
    R.transpose();
    this->fromMatrix(R);


}

template<class Real>
void Quater<Real>::fromMatrix(const defaulttype::Matrix3 &m)
{
    Real tr, s;

    tr = (Real)(m.x().x() + m.y().y() + m.z().z());

    // check the diagonal
    if (tr > 0)
    {
        s = (float)sqrt (tr + 1);
        _q[3] = s * 0.5f; // w OK
        s = 0.5f / s;
        _q[0] = (Real)((m.z().y() - m.y().z()) * s); // x OK
        _q[1] = (Real)((m.x().z() - m.z().x()) * s); // y OK
        _q[2] = (Real)((m.y().x() - m.x().y()) * s); // z OK
    }
    else
    {
        if (m.y().y() > m.x().x() && m.z().z() <= m.y().y())
        {
            s = (Real)sqrt ((m.y().y() - (m.z().z() + m.x().x())) + 1.0f);

            _q[1] = s * 0.5f; // y OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[2] = (Real)((m.y().z() + m.z().y()) * s); // z OK
            _q[0] = (Real)((m.x().y() + m.y().x()) * s); // x OK
            _q[3] = (Real)((m.x().z() - m.z().x()) * s); // w OK
        }
        else if ((m.y().y() <= m.x().x()  &&  m.z().z() > m.x().x())  ||  (m.z().z() > m.y().y()))
        {
            s = (Real)sqrt ((m.z().z() - (m.x().x() + m.y().y())) + 1.0f);

            _q[2] = s * 0.5f; // z OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[0] = (Real)((m.z().x() + m.x().z()) * s); // x OK
            _q[1] = (Real)((m.y().z() + m.z().y()) * s); // y OK
            _q[3] = (Real)((m.y().x() - m.x().y()) * s); // w OK
        }
        else
        {
            s = (Real)sqrt ((m.x().x() - (m.y().y() + m.z().z())) + 1.0f);

            _q[0] = s * 0.5f; // x OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[1] = (Real)((m.x().y() + m.y().x()) * s); // y OK
            _q[2] = (Real)((m.z().x() + m.x().z()) * s); // z OK
            _q[3] = (Real)((m.z().y() - m.y().z()) * s); // w OK
        }
    }
}

// template<class Real> template<class Mat33>
//     void Quater<Real>::toMatrix(Mat33 &m) const
// {
// 	m[0][0] = (1.0 - 2.0 * (_q[1] * _q[1] + _q[2] * _q[2]));
// 	m[0][1] = (2.0 * (_q[0] * _q[1] - _q[2] * _q[3]));
// 	m[0][2] = (2.0 * (_q[2] * _q[0] + _q[1] * _q[3]));
//
// 	m[1][0] = (2.0 * (_q[0] * _q[1] + _q[2] * _q[3]));
// 	m[1][1] = (1.0 - 2.0 * (_q[2] * _q[2] + _q[0] * _q[0]));
// 	m[1][2] = (float) (2.0 * (_q[1] * _q[2] - _q[0] * _q[3]));
//
// 	m[2][0] = (float) (2.0 * (_q[2] * _q[0] - _q[1] * _q[3]));
// 	m[2][1] = (float) (2.0 * (_q[1] * _q[2] + _q[0] * _q[3]));
// 	m[2][2] = (float) (1.0 - 2.0 * (_q[1] * _q[1] + _q[0] * _q[0]));
// }

/// Build a rotation matrix, given a quaternion rotation.
template<class Real>
void Quater<Real>::buildRotationMatrix(Real m[4][4]) const
{
    m[0][0] = (1 - 2 * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[0][1] = (2 * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[0][2] = (2 * (_q[2] * _q[0] + _q[1] * _q[3]));
    m[0][3] = 0;

    m[1][0] = (2 * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1][1] = (1 - 2 * (_q[2] * _q[2] + _q[0] * _q[0]));
    m[1][2] = (2 * (_q[1] * _q[2] - _q[0] * _q[3]));
    m[1][3] = 0;

    m[2][0] = (2 * (_q[2] * _q[0] - _q[1] * _q[3]));
    m[2][1] = (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]));
    m[2][2] = (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]));
    m[2][3] = 0;

    m[3][0] = 0;
    m[3][1] = 0;
    m[3][2] = 0;
    m[3][3] = 1;
}
/// Write an OpenGL rotation matrix
/*template<class Real>
void Quater<Real>::writeOpenGlMatrix(double *m) const
{
    m[0*4+0] = (1.0 - 2.0 * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[0*4+1] = (2.0 * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[0*4+2] = (2.0 * (_q[2] * _q[0] + _q[1] * _q[3]));
    m[0*4+3] = 0.0f;

    m[1*4+0] = (2.0 * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1*4+1] = (1.0 - 2.0 * (_q[2] * _q[2] + _q[0] * _q[0]));
    m[1*4+2] = (float) (2.0 * (_q[1] * _q[2] - _q[0] * _q[3]));
    m[1*4+3] = 0.0f;

    m[2*4+0] = (float) (2.0 * (_q[2] * _q[0] - _q[1] * _q[3]));
    m[2*4+1] = (float) (2.0 * (_q[1] * _q[2] + _q[0] * _q[3]));
    m[2*4+2] = (float) (1.0 - 2.0 * (_q[1] * _q[1] + _q[0] * _q[0]));
    m[2*4+3] = 0.0f;

    m[3*4+0] = 0.0f;
    m[3*4+1] = 0.0f;
    m[3*4+2] = 0.0f;
    m[3*4+3] = 1.0f;
}
*/
/// Write an OpenGL rotation matrix
template<class Real>
void Quater<Real>::writeOpenGlMatrix(double *m) const
{
    m[0*4+0] = (1.0 - 2.0 * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[1*4+0] = (2.0 * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[2*4+0] = (2.0 * (_q[2] * _q[0] + _q[1] * _q[3]));
    m[3*4+0] = 0.0;

    m[0*4+1] = (2.0 * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1*4+1] = (1.0 - 2.0 * (_q[2] * _q[2] + _q[0] * _q[0]));
    m[2*4+1] = (double) (2.0 * (_q[1] * _q[2] - _q[0] * _q[3]));
    m[3*4+1] = 0.0;

    m[0*4+2] = (double) (2.0 * (_q[2] * _q[0] - _q[1] * _q[3]));
    m[1*4+2] = (double) (2.0 * (_q[1] * _q[2] + _q[0] * _q[3]));
    m[2*4+2] = (double) (1.0 - 2.0 * (_q[1] * _q[1] + _q[0] * _q[0]));
    m[3*4+2] = 0.0;

    m[0*4+3] = 0.0;
    m[1*4+3] = 0.0;
    m[2*4+3] = 0.0;
    m[3*4+3] = 1.0;
}

/// Write an OpenGL rotation matrix
template<class Real>
void Quater<Real>::writeOpenGlMatrix(float *m) const
{
    m[0*4+0] = (float) (1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[1*4+0] = (float) (2.0f * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[2*4+0] = (float) (2.0f * (_q[2] * _q[0] + _q[1] * _q[3]));
    m[3*4+0] = 0.0f;

    m[0*4+1] = (float) (2.0f * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1*4+1] = (float) (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]));
    m[2*4+1] = (float) (2.0f * (_q[1] * _q[2] - _q[0] * _q[3]));
    m[3*4+1] = 0.0f;

    m[0*4+2] = (float) (2.0f * (_q[2] * _q[0] - _q[1] * _q[3]));
    m[1*4+2] = (float) (2.0f * (_q[1] * _q[2] + _q[0] * _q[3]));
    m[2*4+2] = (float) (1.0f - 2.0f * (_q[1] * _q[1] + _q[0] * _q[0]));
    m[3*4+2] = 0.0f;

    m[0*4+3] = 0.0f;
    m[1*4+3] = 0.0f;
    m[2*4+3] = 0.0f;
    m[3*4+3] = 1.0f;
}

/// Given an axis and angle, compute quaternion.
template<class Real>
Quater<Real> Quater<Real>::axisToQuat(defaulttype::Vec<3,Real> a, Real phi)
{
    if( a.norm() < std::numeric_limits<Real>::epsilon() )
    {
//		std::cout << "zero norm quaternion" << std::endl;
        _q[0] = _q[1] = _q[2] = (Real)0.0f;
        _q[3] = (Real)1.0f;

        return Quater();
    }

    a = a / a.norm();
    _q[0] = (Real)a.x();
    _q[1] = (Real)a.y();
    _q[2] = (Real)a.z();

    _q[0] = _q[0] * (Real)sin(phi / 2.0);
    _q[1] = _q[1] * (Real)sin(phi / 2.0);
    _q[2] = _q[2] * (Real)sin(phi / 2.0);

    _q[3] = (Real)cos(phi / 2.0);

    return *this;
}

/// Given a quaternion, compute an axis and angle
template<class Real>
void Quater<Real>::quatToAxis(defaulttype::Vec<3,Real> & axis, Real &angle) const
{
    Quater<Real> q = *this;
    if(q[3]<0)
        q*=-1; // we only work with theta in [0, PI]

    Real sin_half_theta; // note that sin(theta/2) == norm of the imaginary part for unit quaternion

    // to avoid numerical instabilities of acos for theta < 5°
    if(q[3]>0.999) // theta < 5° -> q[3] = cos(theta/2) > 0.999
    {
        sin_half_theta = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
        angle = (Real)(2.0 * asin(sin_half_theta));
    }
    else
    {
        Real half_theta = acos(q[3]);
        sin_half_theta = sin(half_theta);
        angle = 2*half_theta;
    }

    assert(sin_half_theta>=0);
    if (sin_half_theta < std::numeric_limits<Real>::epsilon())
        axis = defaulttype::Vec<3,Real>(0.0, 1.0, 0.0);
    else
        axis = defaulttype::Vec<3,Real>(q[0], q[1], q[2])/sin_half_theta;
}

/// Given a quaternion, compute rotation vector (axis times angle)
template<class Real>
defaulttype::Vec<3,Real> Quater<Real>::quatToRotationVector() const
{

    Quater<Real> q = *this;
    q.normalize();

    Real angle;

    if(q[3]<0)
        q*=-1; // we only work with theta in [0, PI] (i.e. angle in [0, 2*PI])

    Real sin_half_theta; // note that sin(theta/2) == norm of the imaginary part for unit quaternion

    // to avoid numerical instabilities of acos for theta < 5°
    if(q[3]>0.999) // theta < 5° -> q[3] = cos(theta/2) > 0.999
    {
        sin_half_theta = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
        angle = (Real)(2.0 * asin(sin_half_theta));
    }
    else
    {
        Real half_theta = acos(q[3]);
        sin_half_theta = sin(half_theta);
        angle = 2*half_theta;
    }

    assert(sin_half_theta>=0);
    defaulttype::Vec<3,Real> rotVector;
    if (sin_half_theta < std::numeric_limits<Real>::epsilon())
        rotVector = defaulttype::Vec<3,Real>(0.0, 0.0, 0.0);
    else
        rotVector = defaulttype::Vec<3,Real>(q[0], q[1], q[2])/sin_half_theta*angle;

    return rotVector;
}


template<class Real>
defaulttype::Vec<3,Real> Quater<Real>::toEulerVector() const
{
///    Compute the Euler angles:
///    Roll: rotation about the X-axis
///    Pitch: rotation about the Y-axis
///    Yaw: rotation about the Z-axis

    Quater<Real> q = *this;
        q.normalize();
        defaulttype::Vec<3,Real> vEuler;
        vEuler[0] = atan2(2*(q[3]*q[0] + q[1]*q[2]) , (1-2*(q[0]*q[0] + q[1]*q[1])));   //roll
        vEuler[1] = asin(2*(q[3]*q[1] - q[2]*q[0]));                                    //pitch
        vEuler[2] = atan2(2*(q[3]*q[2] + q[0]*q[1]) , (1-2*(q[1]*q[1] + q[2]*q[2])));   //yaw
        return vEuler;
}

/*! Returns the slerp interpolation of Quaternions \p a and \p b, at time \p t.

 \p t should range in [0,1]. Result is \p a when \p t=0 and \p b when \p t=1.

 When \p allowFlip is \c true (default) the slerp interpolation will always use the "shortest path"
 between the Quaternions' orientations, by "flipping" the source Quaternion if needed (see
 negate()). */
template<class Real>
void Quater<Real>::slerp(const Quater& a, const Quater& b, Real t, bool allowFlip)
{
    Real cosAngle =  (Real)(a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]);

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
        Real angle    = (Real)acos((Real)std::abs((Real)cosAngle));
        Real sinAngle = (Real)sin((Real)angle);
        c1 = (Real)sin(angle * (1.0f - t)) / sinAngle;
        c2 = (Real)sin(angle * t) / sinAngle;
    }

    // Use the shortest path
    if (allowFlip && (cosAngle < 0.0f))
        c1 = -c1;

    _q[0] = c1*a[0] + c2*b[0];
    _q[1] = c1*a[1] + c2*b[1];
    _q[2] = c1*a[2] + c2*b[2];
    _q[3] = c1*a[3] + c2*b[3];
}

///// Output quaternion
//template<class Real>
//    std::ostream& operator<<(std::ostream& out, Quater<Real> Q)
//{
//	return (out << "(" << Q._q[0] << "," << Q._q[1] << "," << Q._q[2] << ","
//				<< Q._q[3] << ")");
//}

template<class Real>
Quater<Real> Quater<Real>::slerp(Quater<Real> &q1, Real t)
{
    Quater<Real> q0_1;
    for (unsigned int i = 0 ; i<3 ; i++)
        q0_1[i] = -_q[i];

    q0_1[3] = _q[3];

    q0_1 = q1 * q0_1;

    defaulttype::Vec<3,Real> axis, temp;
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
Quater<Real> Quater<Real>::slerp2(Quater<Real> &q1, Real t)
{
    // quaternion to return
    Quater<Real> qm;

    // Calculate angle between them.
    double cosHalfTheta = _q[3] * q1[3] + _q[0] * q1[0] + _q[1] * q1[1] + _q[2] * q1[2];
    // if qa=qb or qa=-qb then theta = 0 and we can return qa
    if (std::abs(cosHalfTheta) >= 1.0)
    {
        qm[3] = _q[3]; qm[0] = _q[0]; qm[1] = _q[1]; qm[2] = _q[2];
        return qm;
    }
    // Calculate temporary values.
    double halfTheta = acos(cosHalfTheta);
    double sinHalfTheta = sqrt(1.0 - cosHalfTheta*cosHalfTheta);
    // if theta = 180 degrees then result is not fully defined
    // we could rotate around any axis normal to qa or qb
    if (std::abs(sinHalfTheta) < 0.001) 
    {
        qm[3] = (Real)(_q[3] * 0.5 + q1[3] * 0.5);
        qm[0] = (Real)(_q[0] * 0.5 + q1[0] * 0.5);
        qm[1] = (Real)(_q[1] * 0.5 + q1[1] * 0.5);
        qm[2] = (Real)(_q[2] * 0.5 + q1[2] * 0.5);
        return qm;
    }
    double ratioA = sin((1 - t) * halfTheta) / sinHalfTheta;
    double ratioB = sin(t * halfTheta) / sinHalfTheta;
    //calculate Quaternion.
    qm[3] = (Real)(_q[3] * ratioA + q1[3] * ratioB);
    qm[0] = (Real)(_q[0] * ratioA + q1[0] * ratioB);
    qm[1] = (Real)(_q[1] * ratioA + q1[1] * ratioB);
    qm[2] = (Real)(_q[2] * ratioA + q1[2] * ratioB);
    return qm;

}

template<class Real>
Quater<Real> Quater<Real>::createQuaterFromFrame(const defaulttype::Vec<3, Real> &lox, const defaulttype::Vec<3, Real> &loy,const defaulttype::Vec<3, Real> &loz)
{
    Quater<Real> q;
    sofa::defaulttype::Mat<3,3, Real> m;

    for (unsigned int i=0 ; i<3 ; i++)
    {
        m[i][0] = lox[i];
        m[i][1] = loy[i];
        m[i][2] = loz[i];
    }
    q.fromMatrix(m);
    return q;
}

/// Print quaternion (C style)
template<class Real>
void Quater<Real>::print()
{
    printf("(%f, %f ,%f, %f)\n", _q[0], _q[1], _q[2], _q[3]);
}

template<class Real>
void Quater<Real>::operator+=(const Quater<Real>& q2)
{
//    static int	count	= 0;

    Real t1[4], t2[4], t3[4];
    Quater<Real> q1 = (*this);
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

/*    if (++count > RENORMCOUNT)
    {
        count = 0;
        normalize();
    } */

	normalize();
}

template<class Real>
void Quater<Real>::operator*=(const Quater<Real>& q1)
{
    Quater<Real> q2 = *this;
    _q[3] = q2._q[3] * q1._q[3] -
            (q2._q[0] * q1._q[0] +
                    q2._q[1] * q1._q[1] +
                    q2._q[2] * q1._q[2]);
    _q[0] = q2._q[3] * q1._q[0] +
            q2._q[0] * q1._q[3] +
            q2._q[1] * q1._q[2] -
            q2._q[2] * q1._q[1];
    _q[1] = q2._q[3] * q1._q[1] +
            q2._q[1] * q1._q[3] +
            q2._q[2] * q1._q[0] -
            q2._q[0] * q1._q[2];
    _q[2] = q2._q[3] * q1._q[2] +
            q2._q[2] * q1._q[3] +
            q2._q[0] * q1._q[1] -
            q2._q[1] * q1._q[0];
}

} // namespace helper

} // namespace sofa

#endif
