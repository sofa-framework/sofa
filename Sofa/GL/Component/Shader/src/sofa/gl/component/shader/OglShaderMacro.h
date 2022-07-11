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

/**
 *  \brief Small class to define macro into an OglShader.
 *
 *  This small abstract class describes macro into an OpenGL shader
 *  (GLSL). It was mainly written for implementing #ifdef macro
 *  into shader, and therefore, to have a multi-purpose shader (and not
 *  many fragmented shaders).
 *
 */

class SOFA_GL_COMPONENT_SHADER_API OglShaderMacro : public OglShaderElement
{
public:
    SOFA_CLASS(OglShaderMacro, OglShaderElement);
protected:
    OglShaderMacro();
    ~OglShaderMacro() override;
public:
    void init() override;

    /// Returns the type of shader element (texture, macro, variable, or attribute)
    ShaderElementType getSEType() const override { return core::visual::ShaderElement::SE_MACRO; }
    // Returns the value of the shader element
    const core::objectmodel::BaseData* getSEValue() const override { return nullptr; }
    // Returns the value of the shader element
    core::objectmodel::BaseData* getSEValue() override { return nullptr; }
};


class SOFA_GL_COMPONENT_SHADER_API OglShaderDefineMacro : public OglShaderMacro
{
public:
    SOFA_CLASS(OglShaderDefineMacro, OglShaderMacro);
protected:
    Data<std::string> value; ///< Set a value for define macro
public:
    OglShaderDefineMacro();
    ~OglShaderDefineMacro() override;
    void init() override;
    // Returns the value of the shader element
    const core::objectmodel::BaseData* getSEValue() const override { return &value; }
    // Returns the value of the shader element
    core::objectmodel::BaseData* getSEValue() override { return &value; }
};

} // namespace sofa::gl::component::shader
