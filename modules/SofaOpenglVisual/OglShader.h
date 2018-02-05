/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "config.h"

#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/visual/Shader.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/gl/template.h>
#include <sofa/helper/gl/GLSLShader.h>
#include <sofa/core/objectmodel/DataFileName.h>

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
 *  or for special treatment that needs shader mechanism.
 *  The 3 kinds of shaders can be defined : vertex, triangle and fragment.
 *  Geometry shader is only available with Nvidia's >8 series
 *  and Ati's >2K series.
 */

class SOFA_OPENGL_VISUAL_API OglShader : public core::visual::Shader, public core::visual::VisualModel
{
public:
    SOFA_CLASS2(OglShader, core::visual::Shader, core::visual::VisualModel);

    ///Activates or not the shader
    Data<bool> turnOn;
    ///Tells if it must be activated automatically(value false : the visitor will switch the shader)
    ///or manually (value true : useful when another component wants to use it for itself only)
    Data<bool> passive;

    ///Files where vertex shader is defined
    sofa::core::objectmodel::DataFileNameVector vertFilename;
    ///Files where fragment shader is defined
    sofa::core::objectmodel::DataFileNameVector fragFilename;
#ifdef GL_GEOMETRY_SHADER_EXT
    ///Files where geometry shader is defined
    sofa::core::objectmodel::DataFileNameVector geoFilename;
#endif

#ifdef GL_TESS_CONTROL_SHADER
    ///Files where tessellation control shader is defined
    sofa::core::objectmodel::DataFileNameVector tessellationControlFilename;
#endif
#ifdef GL_TESS_EVALUATION_SHADER
    ///Files where tessellation evaluation shader is defined
    sofa::core::objectmodel::DataFileNameVector tessellationEvaluationFilename;
#endif

#ifdef GL_GEOMETRY_SHADER_EXT
    ///Describes the input type of primitive if geometry shader is used
    Data<int> geometryInputType;
    ///Describes the output type of primitive if geometry shader is used
    Data<int> geometryOutputType;
    ///Describes the number of vertices in output if geometry shader is used
    Data<int> geometryVerticesOut;
#endif

#ifdef GL_TESS_CONTROL_SHADER
    Data<GLfloat> tessellationOuterLevel;
    Data<GLfloat> tessellationInnerLevel;
#endif

    Data<unsigned int> indexActiveShader;

    // enable writing gl_BackColor in the vertex shader
    Data<bool> backfaceWriting;

    Data<bool> clampVertexColor;

protected:
    ///OpenGL shader
    std::vector<sofa::helper::gl::GLSLShader*> shaderVector;

    OglShader();
    virtual ~OglShader();
public:
    void initVisual() override;
    void init() override;
    void reinit() override;
    void drawVisual(const core::visual::VisualParams* vparams) override;
    void updateVisual() override;
    void parse(core::objectmodel::BaseObjectDescription* arg) override;

    void start() override;
    void stop() override;
    bool isActive() override;

    unsigned int getNumberOfShaders();
    unsigned int getCurrentIndex();
    void setCurrentIndex(const unsigned int index);

    void addDefineMacro(const unsigned int index, const std::string &name, const std::string &value);

    void setTexture(const unsigned int index, const char* name, unsigned short unit);

    void setInt(const unsigned int index, const char* name, int i);
    void setInt2(const unsigned int index, const char* name, int i1, int i2);
    void setInt3(const unsigned int index, const char* name, int i1, int i2, int i3);
    void setInt4(const unsigned int index, const char* name, int i1, int i2, int i3, int i4);

    void setFloat(const unsigned int index, const char* name, float f1);
    void setFloat2(const unsigned int index, const char* name, float f1, float f2);
    void setFloat3(const unsigned int index, const char* name, float f1, float f2, float f3);
    void setFloat4(const unsigned int index, const char* name, float f1, float f2, float f3, float f4);

    void setIntVector(const unsigned int index, const char* name, int count, const GLint* i);
    void setIntVector2(const unsigned int index, const char* name, int count, const GLint* i);
    void setIntVector3(const unsigned int index, const char* name, int count, const GLint* i);
    void setIntVector4(const unsigned int index, const char* name, int count, const GLint* i);

    void setFloatVector(const unsigned int index, const char* name, int count, const float* f);
    void setFloatVector2(const unsigned int index, const char* name, int count, const float* f);
    void setFloatVector3(const unsigned int index, const char* name, int count, const float* f);
    void setFloatVector4(const unsigned int index, const char* name, int count, const float* f);

    void setMatrix2(const unsigned int index, const char* name, int count, bool transpose, const float* f);
    void setMatrix3(const unsigned int index, const char* name, int count, bool transpose, const float* f);
    void setMatrix4(const unsigned int index, const char* name, int count, bool transpose, const float* f);
    void setMatrix2x3(const unsigned int index, const char* name, int count, bool transpose, const float* f);
    void setMatrix3x2(const unsigned int index, const char* name, int count, bool transpose, const float* f);
    void setMatrix2x4(const unsigned int index, const char* name, int count, bool transpose, const float* f);
    void setMatrix4x2(const unsigned int index, const char* name, int count, bool transpose, const float* f);
    void setMatrix3x4(const unsigned int index, const char* name, int count, bool transpose, const float* f);
    void setMatrix4x3(const unsigned int index, const char* name, int count, bool transpose, const float* f);

    GLint getAttribute(const unsigned int index, const char* name);
    GLint getUniform(const unsigned int index, const char* name);

    GLint getGeometryInputType(const unsigned int index) ;
    void  setGeometryInputType(const unsigned int index, GLint v) ;

    GLint getGeometryOutputType(const unsigned int index) ;
    void  setGeometryOutputType(const unsigned int index, GLint v) ;

    GLint getGeometryVerticesOut(const unsigned int index) ;
    void  setGeometryVerticesOut(const unsigned int index, GLint v);


    virtual bool insertInNode( core::objectmodel::BaseNode* node ) override { Inherit1::insertInNode(node); Inherit2::insertInNode(node); return true; }
    virtual bool removeInNode( core::objectmodel::BaseNode* node ) override { Inherit1::removeInNode(node); Inherit2::removeInNode(node); return true; }

};

/**
 *  \brief Abstract class which defines a element to be used with a OglShader.
 *
 *  This is only an partial implementation of the interface ShaderElement
 *  which adds a pointer to its corresponding shader (where it will be used)
 *  and the id (or name) of the element.
 */

class SOFA_OPENGL_VISUAL_API OglShaderElement : public core::visual::ShaderElement
{
protected:
    ///Name of element (corresponding with the shader)
    Data<std::string> id;
    ///Name of element (corresponding with the shader)
    Data<unsigned int> indexShader;
    ///Shader to use the element with
    std::set<OglShader*> shaders;
public:
    OglShaderElement();
    virtual ~OglShaderElement() {}
    virtual void init();
    const std::string getId() const {return id.getValue();}
    void setID( std::string str ) { *(id.beginEdit()) = str; id.endEdit();}
    void setIndexShader( unsigned int index) { *(indexShader.beginEdit()) = index; indexShader.endEdit();}

    // Returns the ID of the shader element
    const std::string& getSEID() const { return id.getValue(); }

    //virtual void setInShader(OglShader& s) = 0;
};

}//namespace visualmodel

} //namespace component

} //namespace sofa

#endif //SOFA_COMPONENT_OGLSHADER
