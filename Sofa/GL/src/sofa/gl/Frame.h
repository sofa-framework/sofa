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
#include <sofa/gl/config.h>

#include <sofa/type/fwd.h>
#include <sofa/type/RGBAColor.h>


namespace sofa::gl
{

class SOFA_GL_API Frame
{
public:
    using Quaternion = sofa::type::Quat<SReal>;

    Frame() = delete;
    ~Frame() = delete;

    static void draw(const type::Vec3& center, const Quaternion& orient, const type::Vec3& length, const type::RGBAColor& colorX = type::RGBAColor::red(), const type::RGBAColor& colorY = type::RGBAColor::green(), const type::RGBAColor& colorZ = type::RGBAColor::red());
    static void draw(const type::Vec3& center, const double orient[4][4], const type::Vec3& length, const type::RGBAColor& colorX = type::RGBAColor::red(), const type::RGBAColor& colorY = type::RGBAColor::green(), const type::RGBAColor& colorZ = type::RGBAColor::red());
    static void draw(const type::Vec3& center, const Quaternion& orient, SReal length = 1.0_sreal, const type::RGBAColor& colorX = type::RGBAColor::red(), const type::RGBAColor& colorY = type::RGBAColor::green(), const type::RGBAColor& colorZ = type::RGBAColor::red());
    static void draw(const type::Vec3& center, const double orient[4][4], SReal length = 1.0_sreal, const type::RGBAColor& colorX = type::RGBAColor::red(), const type::RGBAColor& colorY = type::RGBAColor::green(), const type::RGBAColor& colorZ = type::RGBAColor::red());
   
};

} // namespace sofa::gl
