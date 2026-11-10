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

#include <sofa/defaulttype/fwd.h>
#include <sofa/type/Vec.h>
#include <sofa/type/vector.h>
#include <sofa/helper/random.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraint.h>
#include <istream>
#include <ostream>
#include <algorithm>
#include <memory>
#include <sofa/type/isRigidType.h>


namespace sofa:: defaulttype
{

template<class TCoord, class TDeriv, class TReal>
class StdVectorTypes
{
public:
    typedef TCoord Coord;
    typedef TDeriv Deriv;
    typedef TReal Real;
    typedef type::vector<Coord> VecCoord;
    typedef type::vector<Deriv> VecDeriv;
    typedef type::vector<Real> VecReal;

    static constexpr sofa::Size spatial_dimensions = Coord::spatial_dimensions;
    static constexpr sofa::Size coord_total_size = Coord::total_size;
    static constexpr sofa::Size deriv_total_size = Deriv::total_size;

    typedef typename TCoord::Size Size;

    typedef Coord CPos;
    static constexpr const CPos& getCPos(const Coord& c) { return c; }
    static constexpr void setCPos(Coord& c, const CPos& v) { c = v; }
    typedef Deriv DPos;
    static constexpr const DPos& getDPos(const Deriv& d) { return d; }
    static constexpr void setDPos(Deriv& d, const DPos& v) { d = v; }

    typedef linearalgebra::CompressedRowSparseMatrixConstraint<Deriv> MatrixDeriv;


protected:

    /// @internal size dependent specializations
    /// @{

    template<Size N, class T>
    struct Impl
    {
        static constexpr void set( Coord& c, T x, T y, T z )
        {
            if constexpr (N > 2)
            {
                c[0] = Real(x);
                c[1] = Real(y);
                c[2] = Real(z);
            }
            if constexpr (N == 2)
            {
                SOFA_UNUSED(z);
                c[0] = Real(x);
                c[1] = Real(y);
            }
            if constexpr (N == 1)
            {
                SOFA_UNUSED(y);
                SOFA_UNUSED(z);
                c[0] = Real(x);
            }
        }

        static constexpr void get( T& x, T& y, T& z, const Coord& c )
        {
            if constexpr(N > 2)
            {
                x = T(c[0]);
                y = T(c[1]);
                z = T(c[2]);
            }
            if constexpr (N == 2)
            {
                SOFA_UNUSED(z);
                x = T(c[0]);
                y = T(c[1]);
                z = T(0);
            }
            if constexpr (N == 1)
            {
                SOFA_UNUSED(y);
                SOFA_UNUSED(z);
                x = T(c[0]);
                y = T(0);
                z = T(0);
            }
        }

        static constexpr void add( Coord& c, T x, T y, T z )
        {
            if constexpr (N > 2)
            {
                c[0] += Real(x);
                c[1] += Real(y);
                c[2] += Real(z);
            }
            if constexpr (N == 2)
            {
                SOFA_UNUSED(z);
                c[0] += Real(x);
                c[1] += Real(y);
            }
            if constexpr (N == 1)
            {
                SOFA_UNUSED(y);
                SOFA_UNUSED(z);
                c[0] += Real(x);
            }
        }
    };

    ///@}



public:

    template<typename T>
    static constexpr void set(Coord& c, T x, T y, T z)
    {
        Impl<spatial_dimensions,T>::set(c,x,y,z);
    }

    template<typename T>
    static constexpr void get(T& x, T& y, T& z, const Coord& c)
    {
        Impl<spatial_dimensions,T>::get(x,y,z,c);
    }

    /// Return a Deriv with random value. Each entry with magnitude smaller than the given value.
    static Deriv randomDeriv( Real minMagnitude, Real maxMagnitude )
    {
        Deriv result;
        set( result, Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)), Real(helper::drand(minMagnitude,maxMagnitude)) );
        return result;
    }

    static constexpr Deriv coordDifference(const Coord& c1, const Coord& c2)
    {
        return (Deriv)(c1-c2);
    }

    template<typename T>
    static constexpr void add(Coord& c, T x, T y, T z)
    {
        Impl<spatial_dimensions,T>::add(c,x,y,z);
    }

    static constexpr const char* Name() { return s_name.data();}

    static Coord interpolate(const type::vector< Coord > &ancestors, const type::vector< Real > &coefs)
    {
        assert(ancestors.size() == coefs.size());

        Coord c;

        for (std::size_t i = 0; i < ancestors.size(); i++)
        {
            c += ancestors[i] * coefs[i];
        }

        return c;
    }

private:

    struct Prefix { static constexpr std::string_view value = "Vec"; };

    static constexpr auto s_name = helper::makeFixedSizeTypeName<Prefix, spatial_dimensions, Real>();
};

namespace internal
{
template<sofa::Size N, class T> using VecTypes = StdVectorTypes<type::Vec<N, T>, type::Vec<N, T>, T>;
}

using Vec3dTypes = internal::VecTypes<3, double>;
using Vec2dTypes = internal::VecTypes<2, double>;
using Vec1dTypes = internal::VecTypes<1, double>;
using Vec6dTypes = internal::VecTypes<6, double>;

using Vec3fTypes = internal::VecTypes<3, float>;
using Vec2fTypes = internal::VecTypes<2, float>;
using Vec1fTypes = internal::VecTypes<1, float>;
using Vec6fTypes = internal::VecTypes<6, float>;

using Vec3Types = internal::VecTypes<3, SReal>;
using Vec2Types = internal::VecTypes<2, SReal>;
using Vec1Types = internal::VecTypes<1, SReal>;
using Vec6Types = internal::VecTypes<6, SReal>;


static_assert(std::string_view(Vec1dTypes::Name()) == "Vec1d");
static_assert(std::string_view(Vec2dTypes::Name()) == "Vec2d");
static_assert(std::string_view(Vec3dTypes::Name()) == "Vec3d");

static_assert(std::string_view(Vec1fTypes::Name()) == "Vec1f");
static_assert(std::string_view(Vec2fTypes::Name()) == "Vec2f");
static_assert(std::string_view(Vec3fTypes::Name()) == "Vec3f");

static_assert(!type::isRigidType<Vec6dTypes>);
static_assert(!type::isRigidType<Vec3dTypes>);
static_assert(!type::isRigidType<Vec2dTypes>);
static_assert(!type::isRigidType<Vec1dTypes>);

static_assert(!type::isRigidType<Vec6fTypes>);
static_assert(!type::isRigidType<Vec3fTypes>);
static_assert(!type::isRigidType<Vec2fTypes>);
static_assert(!type::isRigidType<Vec1fTypes>);


} // namespace sofa:: defaulttype
