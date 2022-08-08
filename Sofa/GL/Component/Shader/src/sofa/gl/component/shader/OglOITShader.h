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

#include <sofa/gl/component/shader/config.h>

#include <sofa/gl/component/shader/OglShader.h>

namespace sofa::gl::component::shader
{

class SOFA_GL_COMPONENT_SHADER_API OglOITShader : public sofa::gl::component::shader::OglShader
{
public:
    SOFA_CLASS(OglOITShader, sofa::gl::component::shader::OglShader);
protected:
    OglOITShader();
    ~OglOITShader() override;

public:
    sofa::gl::GLSLShader* accumulationShader();

public:
    static const std::string PATH_TO_OIT_ACCUMULATION_VERTEX_SHADERS;
    static const std::string PATH_TO_OIT_ACCUMULATION_FRAGMENT_SHADERS;

};

} // namespace sofa::gl::component::shader
