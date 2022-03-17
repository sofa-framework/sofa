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

class SOFA_GL_COMPONENT_SHADER_API OglShadowShader : public sofa::gl::component::shader::OglShader
{
public:
    SOFA_CLASS(OglShadowShader, sofa::gl::component::shader::OglShader);
protected:
    OglShadowShader();
    ~OglShadowShader() override;
public:
    void init() override;

    virtual void initShaders(unsigned int numberOfLights, bool softShadow);

protected:
    static const std::string PATH_TO_SHADOW_VERTEX_SHADERS;
    static const std::string PATH_TO_SHADOW_FRAGMENT_SHADERS;
    static const std::string PATH_TO_SOFT_SHADOW_VERTEX_SHADERS;
    static const std::string PATH_TO_SOFT_SHADOW_FRAGMENT_SHADERS;

};

} // namespace sofa::gl::component::shader
