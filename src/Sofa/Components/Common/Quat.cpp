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
Quat::Quat()
{
    _q[0] = _q[1] = _q[2] = _q[3] = 0.0;
}

Quat::Quat(double x, double y, double z, double w)
{
    _q[0] = x;
    _q[1] = y;
    _q[2] = z;
    _q[3] = w;
}

Quat::Quat(double q[])
{
    for (int i = 0; i < 4; i++)
    {
        _q[i] = q[i];
    }
}

// Destructor
Quat::~Quat()
{
}

/// Given two rotations, e1 and e2, expressed as quaternion rotations,
/// figure out the equivalent single rotation and stuff it into dest.
/// This routine also normalizes the result every RENORMCOUNT times it is
/// called, to keep error from creeping in.
///   NOTE: This routine is written so that q1 or q2 may be the same
///  	   as dest (or each other).
Quat operator+(Quat q1, Quat q2)
{
    static int	count	= 0;

    double		t1[4], t2[4], t3[4];
    double		tf[4];
    Quat	ret;

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

    tf[0] = t1[0] + t2[0] + t3[0];
    tf[1] = t1[1] + t2[1] + t3[1];
    tf[2] = t1[2] + t2[2] + t3[2];
    tf[3] = q1._q[3] * q2._q[3] -
            (q1._q[0] * q2._q[0] + q1._q[1] * q2._q[1] + q1._q[2] * q2._q[2]);

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

Quat operator*(const Quat& q1, const Quat& q2)
{
    Quat	ret;

    ret._q[3] = q1._q[3] * q2._q[3] -
            (q1._q[0] * q2._q[0] +
                    q1._q[1] * q2._q[1] +
                    q1._q[2] * q2._q[2]);
    ret._q[0] = q1._q[3] * q2._q[0] +
            q1._q[0] * q2._q[3] +
            q1._q[1] * q2._q[2] -
            q1._q[2] * q2._q[1];
    ret._q[1] = q1._q[3] * q2._q[1] +
            q1._q[1] * q2._q[3] +
            q1._q[2] * q2._q[0] -
            q1._q[0] * q2._q[2];
    ret._q[2] = q1._q[3] * q2._q[2] +
            q1._q[2] * q2._q[3] +
            q1._q[0] * q2._q[1] -
            q1._q[1] * q2._q[0];

    return ret;
}

Quat Quat::quatVectMult(const Vec3d& vect)
{
    Quat	ret;

    ret._q[3] = -(_q[0] * vect[0] + _q[1] * vect[1] + _q[2] * vect[2]);
    ret._q[0] = _q[3] * vect[0] + _q[1] * vect[2] - _q[2] * vect[1];
    ret._q[1] = _q[3] * vect[1] + _q[2] * vect[0] - _q[0] * vect[2];
    ret._q[2] = _q[3] * vect[2] + _q[0] * vect[1] - _q[1] * vect[0];

    return ret;
}

Quat Quat::vectQuatMult(const Vec3d& vect)
{
    Quat	ret;

    ret[3] = -(vect[0] * _q[0] + vect[1] * _q[1] + vect[2] * _q[2]);
    ret[0] = vect[0] * _q[3] + vect[1] * _q[2] - vect[2] * _q[1];
    ret[1] = vect[1] * _q[3] + vect[2] * _q[0] - vect[0] * _q[2];
    ret[2] = vect[2] * _q[3] + vect[0] * _q[1] - vect[1] * _q[0];

    return ret;
}

Quat Quat::inverse()
{
    Quat	ret;

    double		norm	= sqrt(_q[0] * _q[0] +
            _q[1] * _q[1] +
            _q[2] * _q[2] +
            _q[3] * _q[3]);

    if (norm != 0.0)
    {
        norm = 1.0 / norm;
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

/// Quats always obey:  a^2 + b^2 + c^2 + d^2 = 1.0
/// If they don't add up to 1.0, dividing by their magnitude will
/// renormalize them.
void Quat::normalize()
{
    int		i;
    double	mag;

    mag = (_q[0] * _q[0] + _q[1] * _q[1] + _q[2] * _q[2] + _q[3] * _q[3]);
    for (i = 0; i < 4; i++)
    {
        _q[i] /= sqrt(mag);
    }
}

void Quat::fromMatrix(const Mat3x3d &m)
{
    double tr, s;

    tr = m.x().x() + m.y().y() + m.z().z();

    // check the diagonal
    if (tr > 0)
    {
        s = (float)sqrt (tr + 1);
        _q[3] = s * 0.5f; // w OK
        s = 0.5f / s;
        _q[0] = (m.y().z() - m.z().y()) * s; // x OK
        _q[1] = (m.z().x() - m.x().z()) * s; // y OK
        _q[2] = (m.x().y() - m.y().x()) * s; // z OK
    }
    else
    {
        if (m.y().y() > m.x().x() && m.z().z() <= m.y().y())
        {
            s = (float)sqrt ((m.y().y() - (m.z().z() + m.x().x())) + 1.0f);

            _q[1] = s * 0.5f; // y OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[2] = (m.z().y() + m.y().z()) * s; // z OK
            _q[0] = (m.y().x() + m.x().y()) * s; // x OK
            _q[3] = (m.z().x() - m.x().z()) * s; // w OK
        }
        else if ((m.y().y() <= m.x().x()  &&  m.z().z() > m.x().x())  ||  (m.z().z() > m.y().y()))
        {
            s = (float)sqrt ((m.z().z() - (m.x().x() + m.y().y())) + 1.0f);

            _q[2] = s * 0.5f; // z OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[0] = (m.x().z() + m.z().x()) * s; // x OK
            _q[1] = (m.z().y() + m.y().z()) * s; // y OK
            _q[3] = (m.x().y() - m.y().x()) * s; // w OK
        }
        else
        {
            s = (float)sqrt ((m.x().x() - (m.y().y() + m.z().z())) + 1.0f);

            _q[0] = s * 0.5f; // x OK

            if (s != 0.0f)
                s = 0.5f / s;

            _q[1] = (m.y().x() + m.x().y()) * s; // y OK
            _q[2] = (m.x().z() + m.z().x()) * s; // z OK
            _q[3] = (m.y().z() - m.z().y()) * s; // w OK
        }
    }
}

void Quat::toMatrix(Mat3x3d &m) const
{
    m[0][0] = (1.0 - 2.0 * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[0][1] = (2.0 * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[0][2] = (2.0 * (_q[2] * _q[0] + _q[1] * _q[3]));

    m[1][0] = (2.0 * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1][1] = (1.0 - 2.0 * (_q[2] * _q[2] + _q[0] * _q[0]));
    m[1][2] = (float) (2.0 * (_q[1] * _q[2] - _q[0] * _q[3]));

    m[2][0] = (float) (2.0 * (_q[2] * _q[0] - _q[1] * _q[3]));
    m[2][1] = (float) (2.0 * (_q[1] * _q[2] + _q[0] * _q[3]));
    m[2][2] = (float) (1.0 - 2.0 * (_q[1] * _q[1] + _q[0] * _q[0]));
}

/// Build a rotation matrix, given a quaternion rotation.
void Quat::buildRotationMatrix(double m[4][4])
{
    m[0][0] = (1.0 - 2.0 * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[0][1] = (2.0 * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[0][2] = (2.0 * (_q[2] * _q[0] + _q[1] * _q[3]));
    m[0][3] = 0.0f;

    m[1][0] = (2.0 * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1][1] = (1.0 - 2.0 * (_q[2] * _q[2] + _q[0] * _q[0]));
    m[1][2] = (float) (2.0 * (_q[1] * _q[2] - _q[0] * _q[3]));
    m[1][3] = 0.0f;

    m[2][0] = (float) (2.0 * (_q[2] * _q[0] - _q[1] * _q[3]));
    m[2][1] = (float) (2.0 * (_q[1] * _q[2] + _q[0] * _q[3]));
    m[2][2] = (float) (1.0 - 2.0 * (_q[1] * _q[1] + _q[0] * _q[0]));
    m[2][3] = 0.0f;

    m[3][0] = 0.0f;
    m[3][1] = 0.0f;
    m[3][2] = 0.0f;
    m[3][3] = 1.0f;
}

/// Given an axis and angle, compute quaternion.
Quat Quat::axisToQuat(Vec3d a, double phi)
{
    a = a / a.norm();
    _q[0] = a.x();
    _q[1] = a.y();
    _q[2] = a.z();

    _q[0] = _q[0] * sin(phi / 2.0);
    _q[1] = _q[1] * sin(phi / 2.0);
    _q[2] = _q[2] * sin(phi / 2.0);

    _q[3] = cos(phi / 2.0);

    return *this;
}

/// Output quaternion
std::ostream& operator<<(std::ostream& out, Quat Q)
{
    return (out << "(" << Q._q[0] << "," << Q._q[1] << "," << Q._q[2] << ","
            << Q._q[3] << ")");
}

/// Print quaternion (C style)
void Quat::print()
{
    printf("(%f, %f ,%f, %f)\n", _q[0], _q[1], _q[2], _q[3]);
}

void Quat::operator+=(const Quat& q2)
{
    static int	count	= 0;

    double t1[4], t2[4], t3[4];
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

    if (++count > RENORMCOUNT)
    {
        count = 0;
        normalize();
    }
}

void Quat::operator*=(const Quat& q1)
{
    Quat q2 = *this;
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
