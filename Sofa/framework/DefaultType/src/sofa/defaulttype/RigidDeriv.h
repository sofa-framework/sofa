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
#include <sofa/defaulttype/config.h>

#include <sofa/defaulttype/fwd.h>

#include <sofa/type/Vec.h>
#include <sofa/type/Quat.h>
#include <sofa/type/vector.h>
#include <sofa/linearalgebra/matrix_bloc_traits.h>
#include <sofa/helper/rmath.h>
#include <cmath>

#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa::defaulttype
{

template<typename real>
class RigidDeriv<3, real>
{
public:
    typedef real value_type;
    typedef sofa::Size Size;
    typedef real Real;
    typedef type::Vec<3,Real> Pos;
    typedef type::Vec<3,Real> Rot;
    typedef type::Vec<3,Real> Vec3;
    typedef type::Vec<6,Real> VecAll;
    typedef type::Quat<Real> Quat;

protected:
    Vec3 vCenter;
    Vec3 vOrientation;

public:
    friend class RigidCoord<3, real>;

    constexpr RigidDeriv()
    {
        clear();
    }

    explicit constexpr RigidDeriv(type::NoInit)
        : vCenter(type::NOINIT)
        , vOrientation(type::NOINIT)
    {

    }

    constexpr RigidDeriv(const Vec3& velCenter, const Vec3& velOrient)
        : vCenter(velCenter), vOrientation(velOrient)
    {}

    template<typename real2>
    constexpr RigidDeriv(const RigidDeriv<3, real2>& c)
        : vCenter(c.getVCenter()), vOrientation(c.getVOrientation())
    {}

    template<typename real2>
    constexpr RigidDeriv(const type::Vec<6, real2>& v)
        : vCenter(type::Vec<3, real2>(v[0], v[1], v[2])), vOrientation(type::Vec<3, real2>(v[3], v[4], v[5]))
    {}

    template<typename real2>
    constexpr RigidDeriv(const real2* ptr)
        : vCenter(ptr), vOrientation(ptr + 3)
    {
    }

    constexpr void clear()
    {
        vCenter.clear();
        vOrientation.clear();
    }

    template<typename real2>
    constexpr void operator=(const RigidDeriv<3, real2>& c)
    {
        vCenter = c.getVCenter();
        vOrientation = c.getVOrientation();
    }

    template<typename real2>
    constexpr void operator=(const type::Vec<3, real2>& v)
    {
        vCenter = v;
    }

    template<typename real2>
    constexpr void operator=(const type::Vec<6, real2>& v)
    {
        vCenter = v;
        vOrientation = type::Vec<3, real2>(v.data() + 3);
    }

    constexpr void operator+=(const RigidDeriv& a)
    {
        vCenter += a.vCenter;
        vOrientation += a.vOrientation;
    }

    constexpr void operator-=(const RigidDeriv& a)
    {
        vCenter -= a.vCenter;
        vOrientation -= a.vOrientation;
    }

    constexpr RigidDeriv<3, real> operator+(const RigidDeriv<3, real>& a) const
    {
        RigidDeriv d;
        d.vCenter = vCenter + a.vCenter;
        d.vOrientation = vOrientation + a.vOrientation;
        return d;
    }

    template<typename real2>
    constexpr void operator*=(real2 a)
    {
        vCenter *= a;
        vOrientation *= a;
    }

    template<typename real2>
    constexpr void operator/=(real2 a)
    {
        vCenter /= a;
        vOrientation /= a;
    }



    constexpr RigidDeriv<3, real> operator-() const
    {
        return RigidDeriv(-vCenter, -vOrientation);
    }

    constexpr RigidDeriv<3, real> operator-(const RigidDeriv<3, real>& a) const
    {
        return RigidDeriv<3, real>(this->vCenter - a.vCenter, this->vOrientation - a.vOrientation);
    }


    /// dot product, mostly used to compute residuals as sqrt(x*x)
    constexpr Real operator*(const RigidDeriv<3, real>& a) const
    {
        return vCenter[0] * a.vCenter[0] + vCenter[1] * a.vCenter[1] + vCenter[2] * a.vCenter[2]
            + vOrientation[0] * a.vOrientation[0] + vOrientation[1] * a.vOrientation[1]
            + vOrientation[2] * a.vOrientation[2];
    }


    /// Euclidean norm
    real norm() const
    {
        return helper::rsqrt(vCenter * vCenter + vOrientation * vOrientation);
    }


    constexpr Vec3& getVCenter() { return vCenter; }
    constexpr Vec3& getVOrientation() { return vOrientation; }
    constexpr const Vec3& getVCenter() const { return vCenter; }
    constexpr const Vec3& getVOrientation() const { return vOrientation; }

    constexpr Vec3& getLinear() { return vCenter; }
    constexpr  const Vec3& getLinear() const { return vCenter; }
    constexpr Vec3& getAngular() { return vOrientation; }
    constexpr const Vec3& getAngular() const { return vOrientation; }

    constexpr VecAll getVAll() const
    {
        return VecAll(vCenter, vOrientation);
    }

    /// Velocity at point p, where p is the offset from the origin of the frame, given in the same coordinate system as the velocity of the origin.
    constexpr Vec3 velocityAtRotatedPoint(const Vec3& p) const
    {
        return vCenter - cross(p, vOrientation);
    }

    /// write to an output stream
    inline friend std::ostream& operator << (std::ostream& out, const RigidDeriv<3, real>& v)
    {
        out << v.vCenter << " " << v.vOrientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> (std::istream& in, RigidDeriv<3, real>& v)
    {
        in >> v.vCenter >> v.vOrientation;
        return in;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    static constexpr sofa::Size total_size = 6;

    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    static constexpr sofa::Size spatial_dimensions = 3;

    constexpr real* ptr() { return vCenter.ptr(); }
    constexpr const real* ptr() const { return vCenter.ptr(); }

    static constexpr Size size() { return 6; }

    /// Access to i-th element.
    constexpr real& operator[](Size i)
    {
        if (i < 3)
            return this->vCenter(i);
        else
            return this->vOrientation(i - 3);
    }

    /// Const access to i-th element.
    constexpr const real& operator[](Size i) const
    {
        if (i < 3)
            return this->vCenter(i);
        else
            return this->vOrientation(i - 3);
    }

    /// @name Tests operators
    /// @{

    constexpr bool operator==(const RigidDeriv<3, real>& b) const
    {
        return vCenter == b.vCenter && vOrientation == b.vOrientation;
    }

    constexpr bool operator!=(const RigidDeriv<3, real>& b) const
    {
        return vCenter != b.vCenter || vOrientation != b.vOrientation;
    }

    /// @}

};

template<typename real, typename real2>
constexpr RigidDeriv<3,real> operator*(RigidDeriv<3, real> r, real2 a)
{
    r*=a;
    return r;
}

template<typename real, typename real2>
constexpr RigidDeriv<3,real> operator/(RigidDeriv<3, real> r,real2 a)
{
    r/=a;
    return r;
}

template<sofa::Size N,typename T>
constexpr typename RigidDeriv<N,T>::Pos& getLinear(RigidDeriv<N,T>& v)
{
    return v.getLinear();
}

template<sofa::Size N, typename T>
constexpr const typename RigidDeriv<N,T>::Pos& getLinear(const RigidDeriv<N,T>& v)
{
    return v.getLinear();
}

template<sofa::Size N, typename T>
constexpr typename RigidDeriv<N,T>::Rot& getAngular(RigidDeriv<N,T>& v)
{
    return v.getAngular();
}

template<sofa::Size N, typename T>
constexpr const typename RigidDeriv<N,T>::Rot& getAngular(const RigidDeriv<N,T>& v)
{
    return v.getAngular();
}

template<sofa::Size N,typename T>
constexpr typename RigidDeriv<N,T>::Pos& getVCenter(RigidDeriv<N,T>& v)
{
    return v.getLinear();
}

template<sofa::Size N, typename T>
constexpr const typename RigidDeriv<N,T>::Pos& getVCenter(const RigidDeriv<N,T>& v)
{
    return v.getLinear();
}

template<sofa::Size N, typename T>
constexpr typename RigidDeriv<N,T>::Rot& getVOrientation(RigidDeriv<N,T>& v)
{
    return v.getAngular();
}

template<sofa::Size N, typename T>
constexpr const typename RigidDeriv<N,T>::Rot& getVOrientation(const RigidDeriv<N,T>& v)
{
    return v.getAngular();
}

/// Velocity at point p, where p is the offset from the origin of the frame, given in the same coordinate system as the velocity of the origin.
template<typename T, typename R>
constexpr type::Vec<3,T> velocityAtRotatedPoint(const RigidDeriv<3,R>& v, const type::Vec<3,T>& p)
{
    return getLinear(v) + cross( getAngular(v), p );
}

template<typename T, typename R>
constexpr RigidDeriv<3,R> velocityAtRotatedPoint(const RigidDeriv<3,R>& v, const RigidCoord<3,T>& p)
{
    RigidDeriv<3,R> r;
    r.getLinear() = getLinear(v) + cross( getAngular(v), p.getCenter() );
    r.getAngular() = getAngular(v);
    return r;
}

template<typename real>
class RigidDeriv<2,real>
{
public:
    typedef real value_type;
    typedef sofa::Size Size;
    typedef real Real;
    typedef type::Vec<2,Real> Pos;
    typedef Real Rot;
    typedef type::Vec<2,Real> Vec2;
    typedef type::Vec<3,Real> VecAll;

private:
    Vec2 vCenter;
    Real vOrientation;

public:
    friend class RigidCoord<2, real>;

    explicit constexpr RigidDeriv(type::NoInit)
        : vCenter(type::NOINIT)
        , vOrientation(type::NOINIT)
    {

    }

    constexpr RigidDeriv()
    {
        clear();
    }

    template<typename real2>
    constexpr RigidDeriv(const type::Vec<2, real2>& velCenter, const real2& velOrient)
        : vCenter(velCenter), vOrientation((Real)velOrient)
    {}

    template<typename real2>
    constexpr RigidDeriv(const type::Vec<3, real2>& v)
        : vCenter(type::Vec<2, real2>(v.data())), vOrientation((Real)v[2])
    {}

    constexpr void clear()
    {
        vCenter.clear();
        vOrientation = 0;
    }

    template<typename real2>
    constexpr void operator=(const RigidDeriv<2, real2>& c)
    {
        vCenter = c.getVCenter();
        vOrientation = (Real)c.getVOrientation();
    }

    template<typename real2>
    constexpr void operator=(const type::Vec<2, real2>& v)
    {
        vCenter = v;
    }

    template<typename real2>
    constexpr void operator=(const type::Vec<3, real2>& v)
    {
        vCenter = v;
        vOrientation = (Real)v[2];
    }

    constexpr void operator+=(const RigidDeriv<2, real>& a)
    {
        vCenter += a.vCenter;
        vOrientation += a.vOrientation;
    }

    constexpr void operator-=(const RigidDeriv<2, real>& a)
    {
        vCenter -= a.vCenter;
        vOrientation -= a.vOrientation;
    }

    constexpr RigidDeriv<2, real> operator+(const RigidDeriv<2, real>& a) const
    {
        RigidDeriv<2, real> d;
        d.vCenter = vCenter + a.vCenter;
        d.vOrientation = vOrientation + a.vOrientation;
        return d;
    }

    constexpr RigidDeriv<2, real> operator-(const RigidDeriv<2, real>& a) const
    {
        RigidDeriv<2, real> d;
        d.vCenter = vCenter - a.vCenter;
        d.vOrientation = vOrientation - a.vOrientation;
        return d;
    }

    template<typename real2>
    constexpr void operator*=(real2 a)
    {
        vCenter *= a;
        vOrientation *= (Real)a;
    }

    template<typename real2>
    constexpr void operator/=(real2 a)
    {
        vCenter /= a;
        vOrientation /= (Real)a;
    }



    constexpr RigidDeriv<2, real> operator-() const
    {
        return RigidDeriv<2, real>(-vCenter, -vOrientation);
    }

    /// dot product, mostly used to compute residuals as sqrt(x*x)
    constexpr Real operator*(const RigidDeriv<2, real>& a) const
    {
        return vCenter[0] * a.vCenter[0] + vCenter[1] * a.vCenter[1]
            + vOrientation * a.vOrientation;
    }

    /// Euclidean norm
    Real norm() const
    {
        return helper::rsqrt(vCenter * vCenter + vOrientation * vOrientation);
    }

    constexpr Vec2& getVCenter() { return vCenter; }
    constexpr Real& getVOrientation() { return vOrientation; }
    constexpr const Vec2& getVCenter() const { return vCenter; }
    constexpr const Real& getVOrientation() const { return vOrientation; }

    constexpr Vec2& getLinear() { return vCenter; }
    constexpr Real& getAngular() { return vOrientation; }
    constexpr const Vec2& getLinear() const { return vCenter; }
    constexpr const Real& getAngular() const { return vOrientation; }

    constexpr VecAll getVAll() const
    {
        return VecAll(vCenter, vOrientation);
    }

    /// Velocity at point p, where p is the offset from the origin of the frame, given in the same coordinate system as the velocity of the origin.
    constexpr Vec2 velocityAtRotatedPoint(const Vec2& p) const
    {
        return vCenter + Vec2(-p[1], p[0]) * vOrientation;
    }

    /// write to an output stream
    inline friend std::ostream& operator << (std::ostream& out, const RigidDeriv<2, real>& v)
    {
        out << v.vCenter << " " << v.vOrientation;
        return out;
    }
    /// read from an input stream
    inline friend std::istream& operator >> (std::istream& in, RigidDeriv<2, real>& v)
    {
        in >> v.vCenter >> v.vOrientation;
        return in;
    }

    /// Compile-time constant specifying the number of scalars within this vector (equivalent to the size() method)
    static constexpr sofa::Size total_size = 3;

    /// Compile-time constant specifying the number of dimensions of space (NOT equivalent to total_size for rigids)
    static constexpr sofa::Size spatial_dimensions = 2;

    constexpr real* ptr() { return vCenter.ptr(); }
    constexpr const real* ptr() const { return vCenter.ptr(); }

    static constexpr Size size() { return 3; }

    /// Access to i-th element.
    constexpr real& operator[](Size i)
    {
        if (i < 2)
            return this->vCenter(i);
        else
            return this->vOrientation;
    }

    /// Const access to i-th element.
    constexpr const real& operator[](Size i) const
    {
        if (i < 2)
            return this->vCenter(i);
        else
            return this->vOrientation;
    }


    /// @name Tests operators
    /// @{

    constexpr bool operator==(const RigidDeriv<2, real>& b) const
    {
        return vCenter == b.vCenter && vOrientation == b.vOrientation;
    }

    constexpr bool operator!=(const RigidDeriv<2, real>& b) const
    {
        return vCenter != b.vCenter || vOrientation != b.vOrientation;
    }

    /// @}
};

template<typename real, typename real2>
constexpr RigidDeriv<2,real> operator*(RigidDeriv<2,real> r, real2 a)
{
    r *= a;
    return r;
}

template<typename real, typename real2>
constexpr RigidDeriv<2,real> operator/(RigidDeriv<2,real> r, real2 a)
{
    r /= a;
    return r;
}

/// Velocity at point p, where p is the offset from the origin of the frame, given in the same coordinate system as the velocity of the origin.
template<typename R, typename T>
constexpr type::Vec<2,R> velocityAtRotatedPoint(const RigidDeriv<2,T>& v, const type::Vec<2,R>& p)
{
    return getVCenter(v) + type::Vec<2,R>(-p[1], p[0]) * getVOrientation(v);
}

template<typename R, typename T>
constexpr RigidDeriv<2,T> velocityAtRotatedPoint(const RigidDeriv<2,T>& v, const RigidCoord<2,R>& p)
{
    RigidDeriv<2,T> r;
    r.getLinear() = getLinear(v) + type::Vec<2,R>(-p[1], p[0]) * getVOrientation(v);
    r.getAngular() = getAngular(v);
    return r;
}

} // namespace sofa::defaulttype


namespace sofa::linearalgebra
{
    template <Size N, class T, typename IndexType>
    class matrix_bloc_traits < defaulttype::RigidDeriv<N, T>, IndexType >
    {
    public:
        typedef defaulttype::RigidDeriv<N, T> Block;
        typedef T Real;
        typedef Block BlockTranspose;

        enum { NL = 1 };
        enum { NC = defaulttype::RigidDeriv<N, T>::total_size };

        static const Real& v(const Block& b, int /*row*/, int col) { return b[col]; }
        static Real& v(Block& b, int /*row*/, int col) { return b[col]; }
        static void vset(Block& b, int /*row*/, int col, Real v) { b[col] = v; }
        static void vadd(Block& b, int /*row*/, int col, Real v) { b[col] += v; }
        static void clear(Block& b) { b.clear(); }
        static bool empty(const Block& b)
        {
            for (int i = 0; i < NC; ++i)
                if (b[i] != 0) return false;
            return true;
        }

        static BlockTranspose transposed(const Block& b) { return b; }

        static void transpose(BlockTranspose& res, const Block& b) { res = b; }

        template<class TSubBlock, std::enable_if_t<std::is_scalar_v<TSubBlock>, bool> = true>
        static void subBlock(const Block& b, IndexType row, IndexType col, TSubBlock& subBlock)
        {
            SOFA_UNUSED(row);
            subBlock = b[col];
        }

        static sofa::linearalgebra::BaseMatrix::ElementType getElementType()
        {
            return matrix_bloc_traits<Real, IndexType>::getElementType();
        }

        static const char* Name()
        {
            static std::string name = defaulttype::DataTypeName<defaulttype::RigidDeriv<N, T> >::name();
            return name.c_str();
        }
    };
}
