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

/// Build a rotation matrix, given a quaternion rotation.
void Quat::buildRotationMatrix(double m[4][4])
{
    m[0][0] = (float) (1.0 - 2.0 * (_q[1] * _q[1] + _q[2] * _q[2]));
    m[0][1] = (float) (2.0 * (_q[0] * _q[1] - _q[2] * _q[3]));
    m[0][2] = (float) (2.0 * (_q[2] * _q[0] + _q[1] * _q[3]));
    m[0][3] = 0.0f;

    m[1][0] = (float) (2.0 * (_q[0] * _q[1] + _q[2] * _q[3]));
    m[1][1] = (float) (1.0 - 2.0 * (_q[2] * _q[2] + _q[0] * _q[0]));
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
/*
/// Build a rotation matrix, given a quaternion rotation.
void Quat::buildRotationMatrix(Matrix &m)
{
	assert ((m.Col() == 4) && (m.Row() == 4));

	m.set(1, 1, (float) (1.0 - 2.0 * (_q[1] * _q[1] + _q[2] * _q[2])));
	m.set(1, 2, (float) (2.0 * (_q[0] * _q[1] - _q[2] * _q[3])));
	m.set(1, 3, (float) (2.0 * (_q[2] * _q[0] + _q[1] * _q[3])));
	m.set(1, 4, 0.0f);

	m.set(2, 1, (float) (2.0 * (_q[0] * _q[1] + _q[2] * _q[3])));
	m.set(2, 2, (float) (1.0 - 2.0 * (_q[2] * _q[2] + _q[0] * _q[0])));
	m.set(2, 3, (float) (2.0 * (_q[1] * _q[2] - _q[0] * _q[3])));
	m.set(2, 4, 0.0f);

	m.set(3, 1, (float) (2.0 * (_q[2] * _q[0] - _q[1] * _q[3])));
	m.set(3, 2, (float) (2.0 * (_q[1] * _q[2] + _q[0] * _q[3])));
	m.set(3, 3, (float) (1.0 - 2.0 * (_q[1] * _q[1] + _q[0] * _q[0])));
	m.set(3, 4, 0.0f);

	m.set(4, 1, 0.0f);
	m.set(4, 2, 0.0f);
	m.set(4, 3, 0.0f);
	m.set(4, 4, 1.0f);
}
*/

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
