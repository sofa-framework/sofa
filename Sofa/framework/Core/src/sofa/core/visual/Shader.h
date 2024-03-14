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

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa::core::visual
{

/**
 *  \brief A basic interface to define a Shader for different system (OpenGL, DirectX, ...).
 *
 *
 *
 */
class SOFA_CORE_API Shader : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(Shader, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(Shader)
protected:
    Shader() {}
    /// Destructor
    ~Shader() override { }
	
private:
    Shader(const Shader& n) = delete;
    Shader& operator=(const Shader& n) = delete;
	
public:
    /// Start the shader
    virtual void start() = 0;
    /// Stop the shader
    virtual void stop() = 0;
    ///Tells if it must be activated automatically(value false : the visitor will switch the shader)
    ///or manually (value true : useful when another component wants to use it for itself only)
    virtual bool isActive() = 0;

    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;
};

/**
 *  \brief A basic interface to define an element to be used with a Shader.
 */
class SOFA_CORE_API ShaderElement: public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(ShaderElement, objectmodel::BaseObject);
    enum ShaderElementType { SE_NONE = 0, SE_TEXTURE, SE_MACRO, SE_VARIABLE, SE_ATTRIBUTE };
protected:
    ShaderElement() {}
    /// Destructor
    ~ShaderElement() override { }
	
private:
    ShaderElement(const ShaderElement& n) = delete;
    ShaderElement& operator=(const ShaderElement& n) = delete;
	
public:
    /// Returns the type of shader element (texture, macro, variable, or attribute)
    virtual ShaderElementType getSEType() const = 0;
    // Returns the ID of the shader element
    virtual const std::string& getSEID() const = 0;
    // Returns the value of the shader element
    virtual const core::objectmodel::BaseData* getSEValue() const = 0;
    // Returns the value of the shader element
    virtual core::objectmodel::BaseData* getSEValue() = 0;
    // For attributes : return the number of values per vertex
    virtual int getSESizePerVertex() { return 0; }
    // For attributes : return the number of values
    virtual int getSETotalSize() { return 0; }
};
} // namespace sofa::core::visual
