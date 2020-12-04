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

#if __has_include(<sofa/gl/GLSLShader.h>)
#include <sofa/gl/GLSLShader.h>
#define GL_GLSLSHADER_ENABLE_WRAPPER

SOFA_DEPRECATED_HEADER(v21.06, "sofa/gl/GLSLShader.h")

#else
#error "OpenGL headers have been moved to Sofa.GL; you will need to link against this library if you need OpenGL, and include <sofa/gl/GLSLShader.h> instead of this one."
#endif

#ifdef GL_GLSLSHADER_ENABLE_WRAPPER

namespace sofa::helper::gl
{
    using GLSLShader = sofa::gl::GLSLShader;

} // namespace sofa::helper::gl

#endif // GL_AXIS_ENABLE_WRAPPER
