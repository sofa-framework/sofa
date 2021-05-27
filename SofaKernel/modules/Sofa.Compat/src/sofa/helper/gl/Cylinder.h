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

#include <sofa/helper/config.h>

#if __has_include(<sofa/gl/Cylinder.h>)
#include <sofa/gl/Cylinder.h>
#define GL_CYLINDER_ENABLE_WRAPPER

SOFA_DEPRECATED_HEADER(v21.06, "sofa/gl/Cylinder.h")

#else
#error "OpenGL headers have been moved to Sofa.GL. Therefore you will need to link against Sofa.GL if you need OpenGL (PR1649), and include <sofa/gl/Cylinder.h> instead of this one."
#endif

#ifdef GL_CYLINDER_ENABLE_WRAPPER

namespace sofa::helper::gl
{
    using Cylinder = sofa::gl::Cylinder;

} // namespace sofa::helper::gl

#endif // GL_CYLINDER_ENABLE_WRAPPER

#undef GL_CYLINDER_ENABLE_WRAPPER
