#ifndef SOFA_COMPONENTS_COMMON_QUATERNION_H
#define SOFA_COMPONENTS_COMMON_QUATERNION_H

#include "Vec.h"
#include "Mat.h"
#include <assert.h>
#include <iostream>

namespace Sofa
{

namespace Components
{

namespace Common
{

template<class Real>
class Quater
{
private:
    Real _q[4];

public:
    Quater();
    virtual ~Quater();
    Quater(Real x, Real y, Real z, Real w);
    Quater(Real q[]);

    /// Normalize a quaternion
    void normalize();

    void clear()
    {
        _q[0]=0.0;
        _q[1]=0.0;
        _q[2]=0.0;
        _q[3]=1.0;
    }

    void fromMatrix(const Mat3x3d &m);

    template<class Mat33>
    void toMatrix(Mat33 &m) const
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

    /// Apply the rotation to a given vector
    template<class Vec>
    Vec rotate( const Vec& v ) const
    {
        return Vec(
                (1.0 - 2.0 * (_q[1] * _q[1] + _q[2] * _q[2]))*v[0] + (2.0 * (_q[0] * _q[1] - _q[2] * _q[3])) * v[1] + (2.0 * (_q[2] * _q[0] + _q[1] * _q[3])) * v[2],
                (2.0 * (_q[0] * _q[1] + _q[2] * _q[3]))*v[0] + (1.0 - 2.0 * (_q[2] * _q[2] + _q[0] * _q[0]))*v[1] + (2.0 * (_q[1] * _q[2] - _q[0] * _q[3]))*v[2],
                (2.0 * (_q[2] * _q[0] - _q[1] * _q[3]))*v[0] + (2.0 * (_q[1] * _q[2] + _q[0] * _q[3]))*v[1] + (1.0 - 2.0 * (_q[1] * _q[1] + _q[0] * _q[0]))*v[2]
                );

    }

    /// Apply the inverse rotation to a given vector
    template<class Vec>
    Vec inverseRotate( const Vec& v ) const
    {
        return Vec(
                (1.0 - 2.0 * (_q[1] * _q[1] + _q[2] * _q[2]))*v[0] + (2.0 * (_q[0] * _q[1] + _q[2] * _q[3])) * v[1] + (2.0 * (_q[2] * _q[0] - _q[1] * _q[3])) * v[2],
                (2.0 * (_q[0] * _q[1] - _q[2] * _q[3]))*v[0] + (1.0 - 2.0 * (_q[2] * _q[2] + _q[0] * _q[0]))*v[1] + (2.0 * (_q[1] * _q[2] + _q[0] * _q[3]))*v[2],
                (2.0 * (_q[2] * _q[0] + _q[1] * _q[3]))*v[0] + (2.0 * (_q[1] * _q[2] - _q[0] * _q[3]))*v[1] + (1.0 - 2.0 * (_q[1] * _q[1] + _q[0] * _q[0]))*v[2]
                );

    }

    /// Given two quaternions, add them together to get a third quaternion.
    /// Adding quaternions to get a compound rotation is analagous to adding
    /// translations to get a compound translation.
    //template <class T>
    //friend Quater<T> operator+(Quater<T> q1, Quater<T> q2);
    Quater<Real> operator+(const Quater<Real> &q1) const;

    Quater<Real> operator*(const Quater<Real> &q1) const;
    /// Given two Quaters, multiply them together to get a third quaternion.
    //template <class T>
    //friend Quater<T> operator*(const Quater<T>& q1, const Quater<T>& q2);

    Quater quatVectMult(const Vec3d& vect);

    Quater vectQuatMult(const Vec3d& vect);

    Real& operator[](int index)
    {
        assert(index >= 0 && index < 4);
        return _q[index];
    }

    const Real& operator[](int index) const
    {
        assert(index >= 0 && index < 4);
        return _q[index];
    }

    Quater inverse() const;

    // A useful function, builds a rotation matrix in Matrix based on
    // given quaternion.

    void buildRotationMatrix(Real m[4][4]);
    void writeOpenGlMatrix( double* m ) const;

    //void buildRotationMatrix(MATRIX4x4 m);

    //void buildRotationMatrix(Matrix &m);

    // This function computes a quaternion based on an axis (defined by
    // the given vector) and an angle about which to rotate.  The angle is
    // expressed in radians.
    Quater axisToQuat(Vec3d a, Real phi);

    /// Create using rotation vector (axis*angle)
    template<class V>
    static Quater createFromRotationVector(const V& a)
    {
        Real phi = sqrt(a*a);
        if( phi < 1.0e-5 )
            return Quater(0,0,0,1);
        else
        {
            Real nor = 1/phi;
            Real s = sin(phi/2);
            return Quater( a[0]*s*nor, a[1]*s*nor,a[2]*s*nor, cos(phi/2) );
        }
    }


    // Print the quaternion
    template <class T>
    friend std::ostream& operator<<(std::ostream& out, Quater<T> Q);

    // Print the quaternion (C style)
    void print();

    void operator+=(const Quater& q2);
    void operator*=(const Quater& q2);
};

typedef Quater<double> Quat; ///< alias
typedef Quater<double> Quaternion; ///< alias

} // namespace Common

} // namespace Components

} // namespace Sofa

#endif

