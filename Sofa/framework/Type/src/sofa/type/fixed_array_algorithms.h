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

namespace sofa::type::pairwise
{

/// @brief clamp a single value. This function should be removed when std::clamp will be available
template<class T>
const T& stdclamp( const T& v, const T& lo, const T& hi )
{
    assert( !(hi < lo) );
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

/// @brief clamp all the values of a fixed_array to be within a given interval.
template<class T, class TT=typename T::value_type, size_t TN=T::static_size>
T clamp(const T& in, const TT& minValue, const TT& maxValue)
{
    T result {};
    for(typename T::size_type i=0; i < typename T::size_type(TN); ++i)
    {
        result[i] = stdclamp(in[i], minValue, maxValue);
    }
    return result;
}

/// @brief pairwise add of two fixed_array
template<class T, class TT=typename T::value_type, size_t TN=T::static_size>
constexpr T operator+(const T& l, const T& r)
{
    T result {};
    for(typename T::size_type i=0; i < typename T::size_type(TN); ++i)
    {
        result[i] = l[i] + r[i];
    }
    return result;
}

/// @brief pairwise subtract of two fixed_array
template<class T, class TT=typename T::value_type, size_t TN=T::static_size>
constexpr T operator-(const T& l, const T& r)
{
    T result {};
    for(typename T::size_type i=0; i < typename T::size_type(TN); ++i)
    {
        result[i] = l[i] - r[i];
    }
    return result;
}

/// @brief multiply from l the r components.
template<class T, class TT=typename T::value_type, size_t TN=T::static_size>
T operator*(const T& r, const typename T::value_type& f)
{
    T result {};
    for(typename T::size_type i=0; i < typename T::size_type(TN); ++i)
    {
        result[i] = r[i] * f;
    }
    return result;
}

/// @brief multiply from l the r components.
template<class T, class TT=typename T::value_type, size_t TN=T::static_size>
T operator/(const T& r, const typename T::value_type& f)
{
    T result {};
    for(typename T::size_type i=0; i < typename T::size_type(TN); ++i)
    {
        result[i] = r[i] / f;
    }
    return result;
}


} /// namespace sofa::type::pairwise

