#include <math.h>
#include <iostream>
#include "Quat.h"

namespace Sofa
{

namespace Components
{

namespace Common
{

#define RENORMCOUNT 50

// Constructor
template<class Real>
Quater<Real>::Quater()
{
    _q[0] = _q[1] = _q[2] = _q[3] = 0.0;
}

template<class Real>
Quater<Real>::Quater(Real x, Real y, Real z, Real w)
{
    _q[0] = x;
    _q[1] = y;
    _q[2] = z;
    _q[3] = w;
}

template<class Real>
Quater<Real>::Quater( const Vec<3,Real>& axis, Real angle )
{
    axisToQuat(axis,angle);
}

template<class Real>
Quater<Real>::Quater(Real q[])
{
    for (int i = 0; i < 4; i++)
    {
        _q[i] = q[i];
    }
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
    static int	count	= 0;

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

    if (++count > RENORMCOUNT)
    {
        count = 0;
        ret.normalize();
    }

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
Quater<Real> Quater<Real>::quatVectMult(const Vec3d& vect)
{
    Quater<Real>	ret;

    ret._q[3] = (Real) (-(_q[0] * vect[0] + _q[1] * vect[1] + _q[2] * vect[2]));
    ret._q[0] = (Real) (_q[3] * vect[0] + _q[1] * vect[2] - _q[2] * vect[1]);
    ret._q[1] = (Real) (_q[3] * vect[1] + _q[2] * vect[0] - _q[0] * vect[2]);
    ret._q[2] = (Real) (_q[3] * vect[2] + _q[0] * vect[1] - _q[1] * vect[0]);

    return ret;
}

template<class Real>
Quater<Real> Quater<Real>::vectQuatMult(const Vec3d& vect)
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
    int		i;
    Real	mag;

    mag = (_q[0] * _q[0] + _q[1] * _q[1] + _q[2] * _q[2] + _q[3] * _q[3]);
    for (i = 0; i < 4; i++)
    {
        _q[i] /= sqrt(mag);
    }
}

template<class Real>
void Quater<Real>::fromMatrix(const Mat3x3d &m)
{
    Real tr, s;

    tr = (Real)(m.x().x() + m.y().y() + m.z().z());

    // check the diagonal
    if (tr > 0)
    {
        s = (float)sqrt (tr + 1);
        _q[3] = s * 0.5f; // w OK
        s = 0.5f / s;
        _q[0] = (Real)((m.y().z() - m.z().y()) * s); // x OK
        _q[1] = (Real)((m.z().x() - m.x().z()) * s); // y OK
        _q[2] = (Real)((m.x().y() - m.y().x()) * s); // z OK
    }
    else
    {
        if (m.y().y() > m.x().x() && m.z().z() <= m.y().y())
        {
            s = (Real)sqrt ((m.y().y() - (m.z().z() + m.x().x())) + 1.0f);

            _q[1] = s * 0.5f; // y OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[2] = (Real)((m.z().y() + m.y().z()) * s); // z OK
            _q[0] = (Real)((m.y().x() + m.x().y()) * s); // x OK
            _q[3] = (Real)((m.z().x() - m.x().z()) * s); // w OK
        }
        else if ((m.y().y() <= m.x().x()  &&  m.z().z() > m.x().x())  ||  (m.z().z() > m.y().y()))
        {
            s = (Real)sqrt ((m.z().z() - (m.x().x() + m.y().y())) + 1.0f);

            _q[2] = s * 0.5f; // z OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[0] = (Real)((m.x().z() + m.z().x()) * s); // x OK
            _q[1] = (Real)((m.z().y() + m.y().z()) * s); // y OK
            _q[3] = (Real)((m.x().y() - m.y().x()) * s); // w OK
        }
        else
        {
            s = (Real)sqrt ((m.x().x() - (m.y().y() + m.z().z())) + 1.0f);

            _q[0] = s * 0.5f; // x OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[1] = (Real)((m.y().x() + m.x().y()) * s); // y OK
            _q[2] = (Real)((m.x().z() + m.z().x()) * s); // z OK
            _q[3] = (Real)((m.y().z() - m.z().y()) * s); // w OK
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
void Quater<Real>::buildRotationMatrix(Real m[4][4])
{
    m[0][0] = (1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[0][1] = (2.0f * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[0][2] = (2.0f * (_q[2] * _q[0] + _q[1] * _q[3]));
    m[0][3] = 0;

    m[1][0] = (2.0f * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1][1] = (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]));
    m[1][2] = (2.0f * (_q[1] * _q[2] - _q[0] * _q[3]));
    m[1][3] = 0;

    m[2][0] = (2.0f * (_q[2] * _q[0] - _q[1] * _q[3]));
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
    m[0*4+0] = (1.0f - 2.0f * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[1*4+0] = (2.0f * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[2*4+0] = (2.0f * (_q[2] * _q[0] + _q[1] * _q[3]));
    m[3*4+0] = 0.0f;

    m[0*4+1] = (2.0f * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1*4+1] = (1.0f - 2.0f * (_q[2] * _q[2] + _q[0] * _q[0]));
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
Quater<Real> Quater<Real>::axisToQuat(Vec3d a, Real phi)
{
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


///// Output quaternion
//template<class Real>
//    std::ostream& operator<<(std::ostream& out, Quater<Real> Q)
//{
//	return (out << "(" << Q._q[0] << "," << Q._q[1] << "," << Q._q[2] << ","
//				<< Q._q[3] << ")");
//}

/// Print quaternion (C style)
template<class Real>
void Quater<Real>::print()
{
    printf("(%f, %f ,%f, %f)\n", _q[0], _q[1], _q[2], _q[3]);
}

template<class Real>
void Quater<Real>::operator+=(const Quater<Real>& q2)
{
    static int	count	= 0;

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

    if (++count > RENORMCOUNT)
    {
        count = 0;
        normalize();
    }
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


} // namespace Common

} // namespace Components

} // namespace Sofa
