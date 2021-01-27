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

#include <sofa/type/stdtype/fixed_array_algorithms.h>

SOFA_DEPRECATED_HEADER(v21.12, "sofa/type/stdtype/fixed_array_algorithms.h")

namespace sofa::helper::pairwise
{

template<class T>
const T& stdclamp(const T& v, const T& lo, const T& hi)
{
    return sofa::type::stdtype::pairwise::stdclamp(v, lo, hi);
}

template<class T, class TT = typename T::value_type, size_t TN = T::static_size>
T clamp(const T& in, const TT& minValue, const TT& maxValue)
{
    return sofa::type::stdtype::pairwise::clamp(in,minValue,maxValue);
}

template<class T, class TT = typename T::value_type, size_t TN = T::static_size>
T operator+(const T& l, const T& r)
{
    return sofa::type::stdtype::pairwise::operator+(l, r);
}

template<class T, class TT = typename T::value_type, size_t TN = T::static_size>
T operator-(const T& l, const T& r)
{
    return sofa::type::stdtype::pairwise::operator-(l, r);
}

template<class T, class TT = typename T::value_type, size_t TN = T::static_size>
T operator*(const T& r, const typename T::value_type& f)
{
    return sofa::type::stdtype::pairwise::operator*(r, f);
}

template<class T, class TT = typename T::value_type, size_t TN = T::static_size>
T operator/(const T& r, const typename T::value_type& f)
{
    return sofa::type::stdtype::pairwise::operator/(r, f);
}

} // namespace sofa::helper::pairwise
