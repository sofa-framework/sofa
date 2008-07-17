/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
//
// C++ Interface: Shader
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_OGLSHADER
#define SOFA_COMPONENT_OGLSHADER

#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/Shader.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/GLshader.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 *  \brief Utility to use shader for a visual model in OpenGL.
 *
 *  This class is used to implement shader into Sofa, for visual rendering
 *  or for special treatement that needs shader mechanism.
 *  The 3 kinds of shaders can be defined : vertex, triangle and fragment.
 *  Geometry shader is only available with Nvidia's >8 series
 *  and Ati's >2K series.
 */

class OglShader : public core::Shader, public core::VisualModel
{
protected:
    ///Activates or not the shader in live
    Data<bool> turnOn;

    ///File where vertex shader is defined
    Data<std::string> vertFilename;
    ///File where fragment shader is defined
    Data<std::string> fragFilename;
    ///File where geometry shader is defined
    Data<std::string> geoFilename;

    ///Describes the input type of primitive if geometry shader is used
    Data<int> geometryInputType;
    ///Describes the output type of primitive if geometry shader is used
    Data<int> geometryOutputType;
    ///Describes the number of vertices in output if geometry shader is used
    Data<int> geometryVerticesOut;

    ///OpenGL shader
    sofa::helper::gl::CShader m_shader;

    bool hasGeometryShader;

public:
    OglShader();
    virtual ~OglShader();

    void initVisual();
    void init();
    void reinit();
    void drawVisual();
    void updateVisual();

    void start();
    void stop();

    void addDefineMacro(const std::string &name, const std::string &value);

    void setTexture(const char* name, unsigned short unit);

    void setInt(const char* name, int i);
    void setInt2(const char* name, int i1, int i2);
    void setInt3(const char* name, int i1, int i2, int i3);
    void setInt4(const char* name, int i1, int i2, int i3, int i4);

    void setFloat(const char* name, float f1);
    void setFloat2(const char* name, float f1, float f2);
    void setFloat3(const char* name, float f1, float f2, float f3);
    void setFloat4(const char* name, float f1, float f2, float f3, float f4);

    void setIntVector(const char* name, int count, const GLint* i);
    void setIntVector2(const char* name, int count, const GLint* i);
    void setIntVector3(const char* name, int count, const GLint* i);
    void setIntVector4(const char* name, int count, const GLint* i);

    void setFloatVector(const char* name, int count, const float* f);
    void setFloatVector2(const char* name, int count, const float* f);
    void setFloatVector3(const char* name, int count, const float* f);
    void setFloatVector4(const char* name, int count, const float* f);

    GLint getGeometryInputType() ;
    void  setGeometryInputType(GLint v) ;

    GLint getGeometryOutputType() ;
    void  setGeometryOutputType(GLint v) ;

    GLint getGeometryVerticesOut() ;
    void  setGeometryVerticesOut(GLint v);
};

/**
 *  \brief Abstract class which defines a element to be used with a OglShader.
 *
 *  This is only an partial implementation of the interface ShaderElement
 *  which adds a pointer to its corresponding shader (where it will be used)
 *  and the id (or name) of the element.
 */

class OglShaderElement : public core::ShaderElement
{
protected:
    ///Name of element (corresponding with the shader)
    Data<std::string> id;
    ///Shader to use the element with
    OglShader* shader;
public:
    OglShaderElement();
    virtual ~OglShaderElement() { };
    virtual void init();

    //virtual void setInShader(OglShader& s) = 0;
};

}//namespace visualmodel

} //namespace component

} //namespace sofa

#endif //SOFA_COMPONENT_OGLSHADER
