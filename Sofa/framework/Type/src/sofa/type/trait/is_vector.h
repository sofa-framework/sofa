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
#include <type_traits>

namespace sofa::type::trait
{

/// Detect if a type T has iterator/const iterator function, operator[](size_t) and is dynamically resizable (resize function)
template<typename T>
concept is_vector = requires(std::remove_cv_t<T> t, const std::remove_cv_t<T> ct)
{
    t.begin() -> T::iterator;
    t.end() -> T::iterator;

    ct.begin() -> T::const_iterator;
    ct.end() -> T::const_iterator;

    t[0];
    t.resize(1);
};

}
