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

#include <sofa/type/Vec.h>

//SOFA_DEPRECATED_HEADER(v21.12, "sofa/type/Vec.h")

namespace sofa::defaulttype
{
    struct NoInit : public sofa::type::NoInit {};
    constexpr NoInit NOINIT;

    template <std::size_t N, typename real = float>
    using Vec = sofa::type::Vec<N, real>;

    template <std::size_t N, typename real = float>
    using VecNoInit = sofa::type::VecNoInit<N, real>;

    template<typename... Args>
    inline auto dot(Args&&... args) -> decltype(type::dot(std::forward<Args>(args)...))
    { 
        return type::dot(std::forward<Args>(args)...);
    }

    template <typename TVec1, typename TVec2>
    inline auto cross(const TVec1& a, const TVec2& b)
    {
        if constexpr (TVec1::spatial_dimensions == 2 && TVec2::spatial_dimensions == 2)
        {
            return sofa::type::cross(a, b);
        }
        if constexpr (TVec1::spatial_dimensions == 3 && TVec2::spatial_dimensions == 3)
        {
            return sofa::type::cross(a, b);
        }
        else
        {
            static_assert(TVec1::spatial_dimensions == 0, "Cannot call cross with the given type (needs to be Vec2 or Vec3)");
        }
    }



    using Vec1f = sofa::type::Vec1f;
    using Vec1d = sofa::type::Vec1d;
    using Vec1i = sofa::type::Vec1i;
    using Vec1u = sofa::type::Vec1u;
    using Vec1  = sofa::type::Vec1;

    using Vec2f = sofa::type::Vec2f;
    using Vec2d = sofa::type::Vec2d;
    using Vec2i = sofa::type::Vec2i;
    using Vec2u = sofa::type::Vec2u;
    using Vec2 = sofa::type::Vec2;

    using Vec3f = sofa::type::Vec3f;
    using Vec3d = sofa::type::Vec3d;
    using Vec3i = sofa::type::Vec3i;
    using Vec3u = sofa::type::Vec3u;
    using Vec3 = sofa::type::Vec3;

    using Vec4f = sofa::type::Vec4f;
    using Vec4d = sofa::type::Vec4d;
    using Vec4i = sofa::type::Vec4i;
    using Vec4u = sofa::type::Vec4u;
    using Vec4 = sofa::type::Vec4;

    using Vec6f = sofa::type::Vec6f;
    using Vec6d = sofa::type::Vec6d;
    using Vec6i = sofa::type::Vec6i;
    using Vec6u = sofa::type::Vec6u;
    using Vec6 = sofa::type::Vec6;

    using Vector1 = sofa::type::Vector1;
    using Vector2 = sofa::type::Vector2;
    using Vector3 = sofa::type::Vector3;
    using Vector4 = sofa::type::Vector4;
    using Vector6 = sofa::type::Vector6;

} // namespace sofa::defaulttype
