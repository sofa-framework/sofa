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

#include <sofa/types/Vec.h>
#include <sofa/defaulttype/DataTypeInfo.h>

SOFA_PRAGMA_WARNING( \
    This header is deprecated and will be removed at SOFA v21.06.      \
    To fix this warning you must include either sofa/defaulttype/Data_Vec.h if using Vec with Data<> \
    or sofa/types/Vec.h if you do not intend to use Data<> or DataTypeInfo. )

namespace sofa::defaulttype
{
    struct NoInit : public sofa::types::NoInit {};
    constexpr NoInit NOINIT;

    template <std::size_t N, typename real = float>
    using Vec = sofa::types::Vec<N, real>;

    template <std::size_t N, typename real = float>
    using VecNoInit = sofa::types::VecNoInit<N, real>;

    template<typename... Args>
    inline auto dot(Args&&... args) -> decltype(types::dot(std::forward<Args>(args)...))
    { 
        return types::dot(std::forward<Args>(args)...);
    }

    template <typename TVec1, typename TVec2>
    inline auto cross(const TVec1& a, const TVec2& b)
    {
        if constexpr (TVec1::spatial_dimensions == 2 && TVec2::spatial_dimensions == 2)
        {
            return sofa::types::cross(a, b);
        }
        if constexpr (TVec1::spatial_dimensions == 3 && TVec2::spatial_dimensions == 3)
        {
            return sofa::types::cross(a, b);
        }
        else
        {
            static_assert(TVec1::spatial_dimensions == 0, "Cannot call cross with the given types (needs to be Vec2 or Vec3)");
        }
    }



    using Vec1f = sofa::types::Vec1f;
    using Vec1d = sofa::types::Vec1d;
    using Vec1i = sofa::types::Vec1i;
    using Vec1u = sofa::types::Vec1u;
    using Vec1  = sofa::types::Vec1;

    using Vec2f = sofa::types::Vec2f;
    using Vec2d = sofa::types::Vec2d;
    using Vec2i = sofa::types::Vec2i;
    using Vec2u = sofa::types::Vec2u;
    using Vec2 = sofa::types::Vec2;

    using Vec3f = sofa::types::Vec3f;
    using Vec3d = sofa::types::Vec3d;
    using Vec3i = sofa::types::Vec3i;
    using Vec3u = sofa::types::Vec3u;
    using Vec3 = sofa::types::Vec3;

    using Vec4f = sofa::types::Vec4f;
    using Vec4d = sofa::types::Vec4d;
    using Vec4i = sofa::types::Vec4i;
    using Vec4u = sofa::types::Vec4u;
    using Vec4 = sofa::types::Vec4;

    using Vec6f = sofa::types::Vec6f;
    using Vec6d = sofa::types::Vec6d;
    using Vec6i = sofa::types::Vec6i;
    using Vec6u = sofa::types::Vec6u;
    using Vec6 = sofa::types::Vec6;

    using Vector1 = sofa::types::Vector1;
    using Vector2 = sofa::types::Vector2;
    using Vector3 = sofa::types::Vector3;
    using Vector4 = sofa::types::Vector4;
    using Vector6 = sofa::types::Vector6;


// Specialization of the defaulttype::DataTypeInfo type traits template
template<Size N, typename real>
struct DataTypeInfo< sofa::types::Vec<N,real> > : public FixedArrayTypeInfo<sofa::types::Vec<N,real> >
{
    static std::string name() { std::ostringstream o; o << "Vec<" << N << "," << DataTypeName<real>::name() << ">"; return o.str(); }
};

template<Size N, typename real>
struct DataTypeInfo< sofa::types::VecNoInit<N,real> > : public FixedArrayTypeInfo<sofa::types::VecNoInit<N,real> >
{
    static std::string name() { std::ostringstream o; o << "VecNoInit<" << N << "," << DataTypeName<real>::name() << ">"; return o.str(); }
};



// The next line hides all those methods from the doxygen documentation
/// \cond TEMPLATE_OVERRIDES

#define DataTypeInfoName(type,suffix)\
template<Size N>\
struct DataTypeInfo< sofa::types::Vec<N,type> > : public FixedArrayTypeInfo<sofa::types::Vec<N,type> >\
{\
    static std::string name() { std::ostringstream o; o << "Vec" << N << suffix; return o.str(); }\
};\
template<Size N>\
struct DataTypeInfo< sofa::types::VecNoInit<N,type> > : public FixedArrayTypeInfo<sofa::types::VecNoInit<N,type> >\
{\
    static std::string name() { std::ostringstream o; o << "VecNoInit" << N << suffix; return o.str(); }\
};

DataTypeInfoName( float, "f" )
DataTypeInfoName( double, "d" )
DataTypeInfoName( int, "i" )
DataTypeInfoName( unsigned, "u" )

#undef DataTypeInfoName



/// \endcond

} // namespace sofa::defaulttype
