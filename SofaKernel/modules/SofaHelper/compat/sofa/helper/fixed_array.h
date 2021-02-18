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

#include <sofa/type/stdtype/fixed_array.h>

//SOFA_DEPRECATED_HEADER(v21.12, "sofa/type/stdtype/fixed_array.h")

namespace sofa::helper
{
    template<class T, sofa::Size N>
    using fixed_array = sofa::type::stdtype::fixed_array<T, N>;


    template<class T>
    inline fixed_array<T, 2> make_array(const T& v0, const T& v1)
    {
        return sofa::type::stdtype::make_array(v0, v1);
    }

    template<class T>
    inline fixed_array<T, 3> make_array(const T& v0, const T& v1, const T& v2)
    {
        return sofa::type::stdtype::make_array(v0, v1, v2);
    }

    template<class T>
    inline fixed_array<T, 4> make_array(const T& v0, const T& v1, const T& v2, const T& v3)
    {
        return sofa::type::stdtype::make_array(v0, v1, v2, v3);
    }

    template<class T>
    inline fixed_array<T, 5> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4)
    {
        return sofa::type::stdtype::make_array(v0, v1, v2, v3, v4);
    }

    template<class T>
    inline fixed_array<T, 6> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5)
    {
        return sofa::type::stdtype::make_array(v0, v1, v2, v3, v4, v5);
    }

    template<class T>
    inline fixed_array<T, 7> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6)
    {
        return sofa::type::stdtype::make_array(v0, v1, v2, v3, v4, v5, v6);
    }

    template<class T>
    inline fixed_array<T, 8> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6, const T& v7)
    {
        return sofa::type::stdtype::make_array(v0, v1, v2, v3, v4, v5, v6, v7);
    }

    template<class T>
    inline fixed_array<T, 9> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6, const T& v7, const T& v8)
    {
        return sofa::type::stdtype::make_array(v0, v1, v2, v3, v4, v5, v6, v7, v8);
    }

    template<class T>
    inline fixed_array<T, 10> make_array(const T& v0, const T& v1, const T& v2, const T& v3, const T& v4, const T& v5, const T& v6, const T& v7, const T& v8, const T& v9)
    {
        return sofa::type::stdtype::make_array(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9);
    }


} // namespace sofa::helper
