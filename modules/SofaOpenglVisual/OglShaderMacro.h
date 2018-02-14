/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef OGLSHADERMACRO_H_
#define OGLSHADERMACRO_H_
#include "config.h"

#include <SofaOpenglVisual/OglShader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
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

class SOFA_OPENGL_VISUAL_API OglShaderMacro : public OglShaderElement
{
public:
    SOFA_CLASS(OglShaderMacro, OglShaderElement);
protected:
    OglShaderMacro();
    virtual ~OglShaderMacro();
public:
    virtual void init() override;

    /// Returns the type of shader element (texture, macro, variable, or attribute)
    virtual ShaderElementType getSEType() const override { return core::visual::ShaderElement::SE_MACRO; }
    // Returns the value of the shader element
    virtual const core::objectmodel::BaseData* getSEValue() const override { return NULL; }
    // Returns the value of the shader element
    virtual core::objectmodel::BaseData* getSEValue() override { return NULL; }
};


class SOFA_OPENGL_VISUAL_API OglShaderDefineMacro : public OglShaderMacro
{
public:
    SOFA_CLASS(OglShaderDefineMacro, OglShaderMacro);
protected:
    Data<std::string> value; ///< Set a value for define macro
public:
    OglShaderDefineMacro();
    virtual ~OglShaderDefineMacro();
    virtual void init() override;
    // Returns the value of the shader element
    virtual const core::objectmodel::BaseData* getSEValue() const override { return &value; }
    // Returns the value of the shader element
    virtual core::objectmodel::BaseData* getSEValue() override { return &value; }
};

}

}

}

#endif /*OGLSHADERMACRO_H_*/
