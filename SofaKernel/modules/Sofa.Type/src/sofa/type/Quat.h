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

#include <sofa/type/config.h>
#include <sofa/type/fwd.h>

#include <iosfwd>
#include <cassert>

namespace sofa::type
{

template<class Real>
class SOFA_TYPE_API Quat
{
private:
    Real _q[4];

    typedef type::Vec<3, Real> Vec3;
    typedef type::Mat<3,3, Real> Mat3x3;
    typedef type::Mat<4,4, Real> Mat4x4;

public:
    typedef Real value_type;
    typedef sofa::Size Size;

    Quat();
    ~Quat();
    Quat(Real x, Real y, Real z, Real w);

    template<class Real2>
    Quat(const Real2 q[]) { for (int i=0; i<4; i++) _q[i] = Real(q[i]); }

    template<class Real2>
    Quat(const Quat<Real2>& q) { for (int i=0; i<4; i++) _q[i] = Real(q[i]); }

    Quat( const Vec3& axis, Real angle );

    /** Sets this quaternion to the rotation required to rotate direction vector vFrom to direction vector vTo.        
        vFrom and vTo are assumed to be normalized.
    */
    Quat(const Vec3& vFrom, const Vec3& vTo);

    static Quat identity()
    {
        return Quat(0,0,0,1);
    }

    void set(Real x, Real y, Real z, Real w)
    {
        _q[0] = x;
        _q[1] = y;
        _q[2] = z;
        _q[3] = w;
    }

    /// Cast into a standard C array of elements.
    const Real* ptr() const
    {
        return this->_q;
    }

    /// Cast into a standard C array of elements.
    Real* ptr()
    {
        return this->_q;
    }

    /// Returns true if norm of Quaternion is one, false otherwise.
    bool isNormalized();

    /// Normalize a quaternion
    void normalize();

    void clear()
    {
        set(0.0,0.0,0.0,1);
    }

    void fromFrame(Vec3& x, Vec3&y, Vec3&z);
    void fromMatrix(const Mat3x3 &m);

    void toMatrix(Mat3x3 &m) const;
    void toMatrix(Mat4x4 &m) const;

    /// Apply the rotation to a given vector
    auto rotate( const Vec3& v ) const -> Vec3;

    /// Apply the inverse rotation to a given vector
    auto inverseRotate( const Vec3& v ) const -> Vec3;

    /// Given two quaternions, add them together to get a third quaternion.
    /// Adding quaternions to get a compound rotation is analagous to adding
    /// translations to get a compound translation.
    auto operator+(const Quat &q1) const -> Quat;
    auto operator*(const Quat &q1) const -> Quat;

    auto operator*(const Real &r) const -> Quat;
    auto operator/(const Real &r) const -> Quat;
    void operator*=(const Real &r);
    void operator/=(const Real &r);

    /// Given two Quats, multiply them together to get a third quaternion.

    auto quatVectMult(const Vec3& vect) -> Quat;
    auto vectQuatMult(const Vec3& vect) -> Quat;

    Real& operator[](Size index)
    {
        assert(index < 4);
        return _q[index];
    }

    const Real& operator[](Size index) const
    {
        assert(index < 4);
        return _q[index];
    }

    auto inverse() const -> Quat;

    auto quatToRotationVector() const -> Vec3;
    auto toEulerVector() const -> Vec3;

    /*! Returns the slerp interpolation of Quaternions \p a and \p b, at time \p t.
     \p t should range in [0,1]. Result is \p a when \p t=0 and \p b when \p t=1.
     When \p allowFlip is \c true (default) the slerp interpolation will always use the "shortest path"
     between the Quaternions' orientations, by "flipping" the source Quaternion if needed (see
     negate()). */
    void slerp(const Quat& a, const Quat& b, Real t, bool allowFlip=true);

    /// A useful function, builds a rotation matrix in Matrix based on
    /// given quaternion.
    void buildRotationMatrix(Real m[4][4]) const;
    void writeOpenGlMatrix( double* m ) const;
    void writeOpenGlMatrix( float* m ) const;

    /// This function computes a quaternion based on an axis (defined by
    /// the given vector) and an angle about which to rotate.  The angle is
    /// expressed in radians.
    auto axisToQuat(Vec3 a, Real phi) -> Quat;
    void quatToAxis(Vec3 & a, Real &phi) const;

    static auto createQuaterFromFrame(const Vec3 &lox, const Vec3 &loy,const Vec3 &loz) -> Quat;

    /// Create using rotation vector (axis*angle) given in parent coordinates
    static auto createFromRotationVector(const Vec3& a) -> Quat;

    /// Create a quaternion from Euler angles
    /// Thanks to https://github.com/mrdoob/three.js/blob/dev/src/math/Quaternion.js#L199
    enum class EulerOrder
    {
        XYZ, YXZ, ZXY, ZYX, YZX, XZY
    };

    static auto createQuaterFromEuler( Vec3 v, EulerOrder order = EulerOrder::ZYX) -> Quat;

    /// Create a quaternion from Euler angles
    static auto fromEuler( Real alpha, Real beta, Real gamma, EulerOrder order = EulerOrder::ZYX ) -> Quat;

    /// Create using the entries of a rotation vector (axis*angle) given in parent coordinates
    static auto createFromRotationVector(Real a0, Real a1, Real a2 ) -> Quat;

    /// Create using rotation vector (axis*angle) given in parent coordinates
    static auto set(const Vec3& a) { return createFromRotationVector(a); }

    /// Create using using the entries of a rotation vector (axis*angle) given in parent coordinates
    static auto set(Real a0, Real a1, Real a2) { return createFromRotationVector(a0,a1,a2); }

    /// Return the quaternion resulting of the movement between 2 quaternions
    auto quatDiff( Quat a, const Quat& b) -> Quat;

    /// Return the eulerian vector resulting of the movement between 2 quaternions
    auto angularDisplacement( Quat a, const Quat& b) -> Vec3;

    /// Sets this quaternion to the rotation required to rotate direction vector vFrom to direction vector vTo. vFrom and vTo are assumed to be normalized.
    void setFromUnitVectors(const Vec3& vFrom, const Vec3& vTo);

    // Print the quaternion (C style)
    [[deprecated("The function print will be removed soon")]]
    void print();

    auto slerp(Quat &q1, Real t) -> Quat;
    auto slerp2(Quat &q1, Real t) -> Quat;

    void operator+=(const Quat& q2);
    void operator*=(const Quat& q2);
    bool operator==(const Quat& q) const;
    bool operator!=(const Quat& q) const;

    enum { static_size = 4 };
    static unsigned int size() {return 4;}

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    enum { total_size = 4 };
    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for quaternions)
    enum { spatial_dimensions = 3 };
};

/// write to an output stream
template<class Real> SOFA_TYPE_API std::ostream& operator << ( std::ostream& out, const Quat<Real>& v );

/// read from an input stream
template<class Real> SOFA_TYPE_API std::istream& operator >> (std::istream& in, Quat<Real>& v);

#if !defined(SOFA_TYPE_QUAT_CPP)
extern template class SOFA_TYPE_API Quat<double>;
extern template class SOFA_TYPE_API Quat<float>;
#endif

} // namespace sofa::type
