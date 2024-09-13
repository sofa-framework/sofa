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

#include <sofa/type/fixed_array.h>
#include <cstdlib>
#include <functional>
#include <limits>
#include <type_traits>
#include <sofa/type/fwd.h>
#include <cmath>
#include <array>

#define EIGEN_MATRIX_PLUGIN "EigenMatrixAddons.h"
#include <Eigen/Dense>

#define EQUALITY_THRESHOLD 1e-6

namespace sofa::type
{

namespace // anonymous
{
    template<typename real>
    real rabs(const real r)
    {
        if constexpr (std::is_signed<real>())
            return std::abs(r);
        else
            return r;
    }

} // anonymous namespace

//enum NoInit { NOINIT }; ///< use when calling Vec or Mat constructor to skip initialization of values to 0
struct NoInit {};
constexpr NoInit NOINIT;

template < sofa::Size N, typename ValueType>
using Vec = Eigen::Vector<ValueType, N>;


typedef Vec<1,float> Vec1f;
typedef Vec<1,double> Vec1d;
typedef Vec<1,int> Vec1i;
typedef Vec<1,unsigned> Vec1u;
typedef Vec<1,SReal> Vec1;

typedef Vec<2,float> Vec2f;
typedef Vec<2,double> Vec2d;
typedef Vec<2,int> Vec2i;
typedef Vec<2,unsigned> Vec2u;
typedef Vec<2,SReal> Vec2;

typedef Vec<3,float> Vec3f;
typedef Vec<3,double> Vec3d;
typedef Vec<3,int> Vec3i;
typedef Vec<3,unsigned> Vec3u;
typedef Vec<3,SReal> Vec3;

typedef Vec<4,float> Vec4f;
typedef Vec<4,double> Vec4d;
typedef Vec<4,int> Vec4i;
typedef Vec<4,unsigned> Vec4u;
typedef Vec<4,SReal> Vec4;

typedef Vec<6,float> Vec6f;
typedef Vec<6,double> Vec6d;
typedef Vec<6,int> Vec6i;
typedef Vec<6,unsigned> Vec6u;
typedef Vec<6,SReal> Vec6;

template < sofa::Size N, typename ValueType>
auto dot(const Vec<N, ValueType>& a, const Vec<N, ValueType>& b)
{
    return a.dot(b);
}

template < sofa::Size N, typename ValueType>
auto cross(const Vec<N, ValueType>& a, const Vec<N, ValueType>& b)
{
    return a.cross(b);
}

/// Read from an input stream
template<sofa::Size N, typename Real>
std::istream& operator >> ( std::istream& in, Vec<N,Real>& v )
{
    for (sofa::Size i = 0; i < N; ++i)
    {
        in >> v(i);
    }
    return in;
}

/// Write to an output stream
template<sofa::Size N, typename Real>
std::ostream& operator << ( std::ostream& out, const Vec<N,Real>& v )
{
    for (sofa::Size i = 0; i < N - 1; ++i)
    {
        out << v(i) << " ";
    }
    out << v[N - 1];
    return out;
}

} // namespace sofa::type

// Specialization of the std comparison function, to use Vec as std::map key
namespace std
{

// template <>
template<sofa::Size N, class T>
struct less< sofa::type::Vec<N,T> >
{
    bool operator()(const  sofa::type::Vec<N,T>& x, const  sofa::type::Vec<N,T>& y) const
    {
        //msg_info()<<"specialized std::less, x = "<<x<<", y = "<<y<<std::endl;
        for(sofa::Size i=0; i<N; ++i )
        {
            if( x[i]<y[i] )
                return true;
            else if( y[i]<x[i] )
                return false;
        }
        return false;
    }
};

} // namespace std
