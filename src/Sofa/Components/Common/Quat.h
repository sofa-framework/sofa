#ifndef SOFA_COMPONENTS_COMMON_QUATERNION_H
#define SOFA_COMPONENTS_COMMON_QUATERNION_H

#include "Vec.h"
#include <assert.h>
#include <iostream>

namespace Sofa
{

namespace Components
{

namespace Common
{

class Quat
{
private:
    double _q[4];

public:
    Quat();
    virtual ~Quat();
    Quat(double x, double y, double z, double w);
    Quat(double q[]);

    /// Normalize a quaternion
    void normalize();

    void clear() { _q[0]=0.0; _q[1]=0.0; _q[2]=0.0; _q[3]=1.0; }

    /// Given two quaternions, add them together to get a third quaternion.
    /// Adding quaternions to get a compound rotation is analagous to adding
    /// translations to get a compound translation.
    friend Quat operator+(Quat q1, Quat q2);

    /// Given two Quats, multiply them together to get a third quaternion.
    friend Quat operator*(const Quat& q1, const Quat& q2);

    Quat quatVectMult(const Vec3d& vect);

    Quat vectQuatMult(const Vec3d& vect);

    double& operator[](int index)
    {
        assert(index >= 0 && index < 4);
        return _q[index];
    }

    const double& operator[](int index) const
    {
        assert(index >= 0 && index < 4);
        return _q[index];
    }

    Quat inverse();

    // A useful function, builds a rotation matrix in Matrix based on
    // given quaternion.

    void buildRotationMatrix(double m[4][4]);

    //void buildRotationMatrix(MATRIX4x4 m);

    //void buildRotationMatrix(Matrix &m);

    // This function computes a quaternion based on an axis (defined by
    // the given vector) and an angle about which to rotate.  The angle is
    // expressed in radians.
    Quat axisToQuat(Vec3d a, double phi);

    // Print the quaternion
    friend std::ostream& operator<<(std::ostream& out, Quat Q);

    // Print the quaternion (C style)
    void print();

    void operator+=(const Quat& q2);
    void operator*=(const Quat& q2);
};

typedef Quat Quaternion; ///< alias

} // namespace Common

} // namespace Components

} // namespace Sofa

#endif
