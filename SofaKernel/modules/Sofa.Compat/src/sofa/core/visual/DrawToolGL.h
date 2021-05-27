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


#if __has_include(<sofa/gl/DrawToolGL.h>)
#include <sofa/gl/DrawToolGL.h>
#define GL_DRAWTOOLGL_ENABLE_WRAPPER

SOFA_DEPRECATED_HEADER(v21.06, "sofa/gl/DrawToolGL.h")

#else
#error "OpenGL headers have been moved to Sofa.GL. Therefore you will need to link against Sofa.GL if you need OpenGL (PR1649), and include <sofa/gl/DrawToolGL.h> instead of this one."
#endif

#ifdef GL_DRAWTOOLGL_ENABLE_WRAPPER

namespace sofa::core::visual
{
    using DrawToolGL = sofa::gl::DrawToolGL;

} // namespace sofa::core::visual

#endif // GL_DRAWTOOLGL_ENABLE_WRAPPER

#undef GL_DRAWTOOLGL_ENABLE_WRAPPER

