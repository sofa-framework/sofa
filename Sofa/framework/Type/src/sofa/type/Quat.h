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

#include <sofa/type/fixed_array.h>

#include <iosfwd>
#include <cassert>

namespace // anonymous
{

template<typename QuatReal, typename OtherReal>
constexpr void getOpenGlMatrix(const QuatReal& q, OtherReal* m)
{
    m[0 * 4 + 0] = static_cast<OtherReal>(1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
    m[1 * 4 + 0] = static_cast<OtherReal>(2.0 * (q[0] * q[1] - q[2] * q[3]));
    m[2 * 4 + 0] = static_cast<OtherReal>(2.0 * (q[2] * q[0] + q[1] * q[3]));
    m[3 * 4 + 0] = static_cast<OtherReal>(0.0);

    m[0 * 4 + 1] = static_cast<OtherReal>(2.0 * (q[0] * q[1] + q[2] * q[3]));
    m[1 * 4 + 1] = static_cast<OtherReal>(1.0 - 2.0 * (q[2] * q[2] + q[0] * q[0]));
    m[2 * 4 + 1] = static_cast<OtherReal>(2.0 * (q[1] * q[2] - q[0] * q[3]));
    m[3 * 4 + 1] = static_cast<OtherReal>(0.0);

    m[0 * 4 + 2] = static_cast<OtherReal>(2.0 * (q[2] * q[0] - q[1] * q[3]));
    m[1 * 4 + 2] = static_cast<OtherReal>(2.0 * (q[1] * q[2] + q[0] * q[3]));
    m[2 * 4 + 2] = static_cast<OtherReal>(1.0 - 2.0 * (q[1] * q[1] + q[0] * q[0]));
    m[3 * 4 + 2] = static_cast<OtherReal>(0.0);

    m[0 * 4 + 3] = static_cast<OtherReal>(0.0);
    m[1 * 4 + 3] = static_cast<OtherReal>(0.0);
    m[2 * 4 + 3] = static_cast<OtherReal>(0.0);
    m[3 * 4 + 3] = static_cast<OtherReal>(1.0);
}

}

namespace sofa::type
{

struct qNoInit {};
constexpr qNoInit QNOINIT;

template<class Real>
class Quat
{
    sofa::type::fixed_array<Real, 4> _q{};

    typedef type::Vec<3, Real> Vec3;
    typedef type::Mat<3,3, Real> Mat3x3;
    typedef type::Mat<4,4, Real> Mat4x4;

public:
    typedef Real value_type;
    typedef sofa::Size Size;

    constexpr Quat()
    {
        this->clear();
    }

    /// Fast constructor: no initialization
    explicit constexpr Quat(qNoInit)
    {
    }

    ~Quat() = default;
    constexpr Quat(Real x, Real y, Real z, Real w)
    {
        set(x, y, z, w);
    }

    template<class Real2>
    constexpr Quat(const Real2 q[])
    { 
        for (int i=0; i<4; i++) 
            _q[i] = Real(q[i]); 
    }

    template<class Real2>
    constexpr Quat(const Quat<Real2>& q) 
    { 
        for (int i=0; i<4; i++) 
            _q[i] = Real(q[i]); 
    }

    Quat( const Vec3& axis, Real angle );

    /** Sets this quaternion to the rotation required to rotate direction vector vFrom to direction vector vTo.        
        vFrom and vTo are assumed to be normalized.
    */
    Quat(const Vec3& vFrom, const Vec3& vTo);

    static Quat identity()
    {
        return Quat(0, 0, 0, 1);
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
        return this->_q.data();
    }

    /// Cast into a standard C array of elements.
    Real* ptr()
    {
        return &(this->_q[0]);
    }

    /// Returns true if norm of Quaternion is one, false otherwise.
    bool isNormalized();

    /// Normalize a quaternion
    void normalize();

    void clear()
    {
        set(0, 0, 0, 1);
    }

    /// Convert the reference frame orientation into an orientation quaternion
    void fromFrame(const Vec3& x, const Vec3&y, const Vec3&z);

    /// Convert a rotation matrix into an orientation quaternion
    void fromMatrix(const Mat3x3 &m);

    /// Convert the quaternion into an orientation matrix
    void toMatrix(Mat3x3 &m) const
    {
        m(0,0) = (1 - 2 * (_q[1] * _q[1] + _q[2] * _q[2]));
        m(0,1) = (2 * (_q[0] * _q[1] - _q[2] * _q[3]));
        m(0,2) = (2 * (_q[2] * _q[0] + _q[1] * _q[3]));

        m(1,0) = (2 * (_q[0] * _q[1] + _q[2] * _q[3]));
        m(1,1) = (1 - 2 * (_q[2] * _q[2] + _q[0] * _q[0]));
        m(1,2) = (2 * (_q[1] * _q[2] - _q[0] * _q[3]));

        m(2,0) = (2 * (_q[2] * _q[0] - _q[1] * _q[3]));
        m(2,1) = (2 * (_q[1] * _q[2] + _q[0] * _q[3]));
        m(2,2) = (1 - 2 * (_q[1] * _q[1] + _q[0] * _q[0]));
    }

    /// Convert the quaternion into an orientation homogeneous matrix
    /// The homogeneous part is set to 0,0,0,1
    constexpr void toHomogeneousMatrix(Mat4x4 &m) const
    {
        m(0,0) = (1 - 2 * (_q[1] * _q[1] + _q[2] * _q[2]));
        m(0,1) = (2 * (_q[0] * _q[1] - _q[2] * _q[3]));
        m(0,2) = (2 * (_q[2] * _q[0] + _q[1] * _q[3]));
        m(0,3) = 0.0;

        m(1,0) = (2 * (_q[0] * _q[1] + _q[2] * _q[3]));
        m(1,1) = (1 - 2 * (_q[2] * _q[2] + _q[0] * _q[0]));
        m(1,2) = (2 * (_q[1] * _q[2] - _q[0] * _q[3]));
        m(1,3) = 0.0;

        m(2,0) = (2 * (_q[2] * _q[0] - _q[1] * _q[3]));
        m(2,1) = (2 * (_q[1] * _q[2] + _q[0] * _q[3]));
        m(2,2) = (1 - 2 * (_q[1] * _q[1] + _q[0] * _q[0]));
        m(2,3) = 0.0;

        m(3,0) = 0.0f;
        m(3,1) = 0.0f;
        m(3,2) = 0.0f;
        m(3,3) = 1.0f;
    }

    /// Apply the rotation to a given vector
    constexpr auto rotate( const Vec3& v ) const -> Vec3
    {
        const Vec3 qxyz{ _q[0], _q[1] , _q[2] };
        const auto t = qxyz.cross(v) * 2;
        return (v + _q[3] * t + qxyz.cross(t));
    }

    /// Apply the inverse rotation to a given vector
    constexpr auto inverseRotate( const Vec3& v ) const -> Vec3
    {
        const Vec3 qxyz{ -_q[0], -_q[1] , -_q[2] };
        const auto t = qxyz.cross(v) * 2;
        return (v + _q[3] * t + qxyz.cross(t));
    }

    /// Given two quaternions, add them together to get a third quaternion.
    /// Adding quaternions to get a compound rotation is analogous to adding
    /// translations to get a compound translation.
    auto operator+(const Quat &q1) const -> Quat;
    constexpr auto operator*(const Quat& q1) const -> Quat
    {
        Quat	ret(QNOINIT);

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

    constexpr auto operator*(const Real &r) const -> Quat
    {
        Quat  ret(QNOINIT);
        ret[0] = _q[0] * r;
        ret[1] = _q[1] * r;
        ret[2] = _q[2] * r;
        ret[3] = _q[3] * r;
        return ret;
    }

    auto operator/(const Real &r) const -> Quat
    {
        Quat  ret(QNOINIT);
        ret[0] = _q[0] / r;
        ret[1] = _q[1] / r;
        ret[2] = _q[2] / r;
        ret[3] = _q[3] / r;
        return ret;
    }

    void operator*=(const Real &r)
    {
        _q[0] *= r;
        _q[1] *= r;
        _q[2] *= r;
        _q[3] *= r;
    }

    void operator/=(const Real &r)
    {
        _q[0] /= r;
        _q[1] /= r;
        _q[2] /= r;
        _q[3] /= r;
    }

    /// Given two Quats, multiply them together to get a third quaternion.
    constexpr auto quatVectMult(const Vec3& vect) const -> Quat
    {
        Quat ret(QNOINIT);
        ret._q[3] = -(_q[0] * vect[0] + _q[1] * vect[1] + _q[2] * vect[2]);
        ret._q[0] = _q[3] * vect[0] + _q[1] * vect[2] - _q[2] * vect[1];
        ret._q[1] = _q[3] * vect[1] + _q[2] * vect[0] - _q[0] * vect[2];
        ret._q[2] = _q[3] * vect[2] + _q[0] * vect[1] - _q[1] * vect[0];

        return ret;
    }

    constexpr auto vectQuatMult(const Vec3& vect) const -> Quat
    {
        Quat ret(QNOINIT);
        ret[3] = -(vect[0] * _q[0] + vect[1] * _q[1] + vect[2] * _q[2]);
        ret[0] = vect[0] * _q[3] + vect[1] * _q[2] - vect[2] * _q[1];
        ret[1] = vect[1] * _q[3] + vect[2] * _q[0] - vect[0] * _q[2];
        ret[2] = vect[2] * _q[3] + vect[0] * _q[1] - vect[1] * _q[0];
        return ret;
    }

    constexpr Real& operator[](Size index)
    {
        assert(index < 4);
        return _q[index];
    }

    constexpr const Real& operator[](Size index) const
    {
        assert(index < 4);
        return _q[index];
    }

    template< std::size_t I >
    [[nodiscard]] constexpr Real& get() & noexcept requires (I < 4)
    {
        return _q[I];
    }

    template< std::size_t I >
    [[nodiscard]] constexpr const Real& get() const& noexcept requires (I < 4)
    {
        return _q[I];
    }

    template< std::size_t I >
    [[nodiscard]] constexpr Real&& get() && noexcept requires (I < 4)
    {
        return std::move(_q[I]);
    }

    template< std::size_t I >
    [[nodiscard]] constexpr const Real&& get() const&& noexcept requires (I < 4)
    {
        return std::move(_q[I]);
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
    constexpr void buildRotationMatrix(Real m[4][4]) const
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

    constexpr void writeOpenGlMatrix(double* m) const
    {
        return getOpenGlMatrix<Quat, double>(*this, m);
    }

    constexpr void writeOpenGlMatrix(float* m) const
    {
        return getOpenGlMatrix<Quat, float>(*this, m);
    }

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

    static auto createQuaterFromEuler(const Vec3& v, EulerOrder order = EulerOrder::ZYX) -> Quat;

    /// Create a quaternion from Euler angles
    static auto fromEuler( Real alpha, Real beta, Real gamma, EulerOrder order = EulerOrder::ZYX ) -> Quat;

    /// Create using the entries of a rotation vector (axis*angle) given in parent coordinates
    static auto createFromRotationVector(Real a0, Real a1, Real a2 ) -> Quat;

    /// Create using rotation vector (axis*angle) given in parent coordinates
    static auto set(const Vec3& a) { return createFromRotationVector(a); }

    /// Create using using the entries of a rotation vector (axis*angle) given in parent coordinates
    static auto set(Real a0, Real a1, Real a2) { return createFromRotationVector(a0,a1,a2); }

    /// Return the quaternion resulting of the movement between 2 quaternions
    static auto quatDiff( Quat a, const Quat& b) -> Quat;

    /// Return the eulerian vector resulting of the movement between 2 quaternions
    static auto angularDisplacement( const Quat& a, const Quat& b) -> Vec3;

    /// Sets this quaternion to the rotation required to rotate direction vector vFrom to direction vector vTo. vFrom and vTo are assumed to be normalized.
    void setFromUnitVectors(const Vec3& vFrom, const Vec3& vTo);

    auto slerp(const Quat &q1, Real t) const -> Quat;
    auto slerp2(const Quat &q1, Real t) const-> Quat;

    void operator+=(const Quat& q2);
    constexpr void operator*=(const Quat& q1)
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

    bool operator==(const Quat& q) const;
    bool operator!=(const Quat& q) const;

    static constexpr Size static_size = 4;
    static Size size() {return static_size;}

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    static constexpr Size total_size = 4;

    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for quaternions)
    static constexpr Size spatial_dimensions = 3;
};


/// Same as Quat except the values are not initialized by default
template<class Real>
class QuatNoInit : public Quat<Real>
{
public:
    constexpr QuatNoInit() noexcept
        : Quat<Real>(QNOINIT)
    {}
    using Quat<Real>::Quat;

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

namespace std
{

template<class Real>
struct tuple_size<::sofa::type::Quat<Real> > : integral_constant<size_t, 4> {};

template<std::size_t I, class Real>
struct tuple_element<I, ::sofa::type::Quat<Real> >
{
    using type = typename::sofa::type::Quat<Real>::value_type;
};

}
