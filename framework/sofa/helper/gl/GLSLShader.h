/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_GL_GLSLSHADER_H
#define SOFA_HELPER_GL_GLSLSHADER_H

#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>

#include <string>
#include <string.h>

#include <sofa/helper/helper.h>
#include <vector>
#include <iostream>

namespace sofa
{

namespace helper
{

namespace gl
{

#ifndef SOFA_HAVE_GLEW
#error GL Shader support requires GLEW. Please define SOFA_HAVE_GLEW to use shaders.
#endif

class SOFA_HELPER_API GLSLShader
{
public:

    GLSLShader();
    ~GLSLShader();

    /// This builds a header before any shader contents
    void AddHeader(const std::string& header);

    void AddDefineMacro(const std::string &name, const std::string &value);

    /// This loads our text file for each shader and returns it in a string
    std::string LoadTextFile(const std::string& strFile);

    /// This is used to load all of the extensions and checks compatibility.
    static bool InitGLSL();

    /// This loads a vertex, geometry and fragment shader
    void InitShaders(const std::string& strVertex, const std::string& stdGeometry, const std::string& strFragment);

    /// This loads a vertex and fragment shader
    void InitShaders(const std::string& strVertex, const std::string& strFragment)
    {
        InitShaders(strVertex, std::string(""), strFragment);
    }

    /// This returns an ID for a variable in our shader
    GLint GetVariable(std::string strVariable);

    /// This returns an ID for an attribute variable in our shader
    GLint GetAttributeVariable(std::string strVariable);

    /// These are our basic get functions for our private data
    /// @{
    GLhandleARB GetProgram()	{	return m_hProgramObject; }
    GLhandleARB GetVertexS()	{	return m_hVertexShader; }
    GLhandleARB GetGeometryS()	{	return m_hGeometryShader; }
    GLhandleARB GetFragmentS()	{	return m_hFragmentShader; }

    /// Below are functions to set an integer or a float
    /// @{
    void SetInt(GLint variable, int newValue);
    void SetFloat(GLint variable, float newValue);
    /// @}

    /// Below are functions to set more than 1 integer or float
    /// @{
    void SetInt2(GLint variable, int i1, int i2);
    void SetInt3(GLint variable, int i1, int i2, int i3);
    void SetInt4(GLint variable, int i1, int i2, int i3, int i4);
    void SetFloat2(GLint variable, float v0, float v1);
    void SetFloat3(GLint variable, float v0, float v1, float v2);
    void SetFloat4(GLint variable, float v0, float v1, float v2, float v3);
    /// @}

    /// Below are functions to set a vector of integer or float
    /// @{
    void SetIntVector(GLint variable, GLsizei count, const GLint *value);
    void SetIntVector2(GLint variable, GLsizei count, const GLint *value);
    void SetIntVector3(GLint variable, GLsizei count, const GLint *value);
    void SetIntVector4(GLint variable, GLsizei count, const GLint *value);

    void SetFloatVector(GLint variable, GLsizei count, const float *value);
    void SetFloatVector2(GLint variable, GLsizei count, const float *value);
    void SetFloatVector3(GLint variable, GLsizei count, const float *value);
    void SetFloatVector4(GLint variable, GLsizei count, const float *value);
    /// @}

    /// Below are functions to set a matrix
    /// @{
    void SetMatrix2(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    void SetMatrix3(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    void SetMatrix4(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);

#ifdef GL_VERSION_2_1
    void SetMatrix2x3(GLint location,GLsizei count,GLboolean transpose, const GLfloat *value);
    void SetMatrix3x2(GLint location,GLsizei count,GLboolean transpose, const GLfloat *value);
    void SetMatrix2x4(GLint location,GLsizei count,GLboolean transpose, const GLfloat *value);
    void SetMatrix4x2(GLint location,GLsizei count,GLboolean transpose, const GLfloat *value);
    void SetMatrix3x4(GLint location,GLsizei count,GLboolean transpose, const GLfloat *value);
    void SetMatrix4x3(GLint location,GLsizei count,GLboolean transpose, const GLfloat *value);
#else
    void SetMatrix2x3(GLint /*location*/,GLsizei /*count*/,GLboolean /*transpose*/, const GLfloat */*value*/) { fprintf(stderr,"SetMatrix2x3 not supported ."); }
    void SetMatrix3x2(GLint /*location*/,GLsizei /*count*/,GLboolean /*transpose*/, const GLfloat */*value*/) { fprintf(stderr,"SetMatrix3x2 not supported ."); }
    void SetMatrix2x4(GLint /*location*/,GLsizei /*count*/,GLboolean /*transpose*/, const GLfloat */*value*/) { fprintf(stderr,"SetMatrix2x4 not supported ."); }
    void SetMatrix4x2(GLint /*location*/,GLsizei /*count*/,GLboolean /*transpose*/, const GLfloat */*value*/) { fprintf(stderr,"SetMatrix4x2 not supported ."); }
    void SetMatrix3x4(GLint /*location*/,GLsizei /*count*/,GLboolean /*transpose*/, const GLfloat */*value*/) { fprintf(stderr,"SetMatrix3x4 not supported ."); }
    void SetMatrix4x3(GLint /*location*/,GLsizei /*count*/,GLboolean /*transpose*/, const GLfloat */*value*/) { fprintf(stderr,"SetMatrix4x3 not supported ."); }
#endif
    /// @}


    /// These 2 functions turn on and off our shader
    /// @{
    void TurnOn();
    void TurnOff();
    /// @}

    /// This releases our memory for our shader
    void Release();

    GLint GetGeometryInputType() { return geometry_input_type; }
    void  SetGeometryInputType(GLint v) { geometry_input_type = v; }

    GLint GetGeometryOutputType() { return geometry_output_type; }
    void  SetGeometryOutputType(GLint v) { geometry_output_type = v; }

    GLint GetGeometryVerticesOut() { return geometry_vertices_out; }
    void  SetGeometryVerticesOut(GLint v) { geometry_vertices_out = v; }

protected:

    bool CompileShader(GLint target, const std::string& fileName, const std::string& header, GLhandleARB& shader);

    std::string header;

    /// This handle stores our vertex shader information
    GLhandleARB m_hVertexShader;

    /// This handle stores our geometry shader information
    GLhandleARB m_hGeometryShader;

    /// This handle stores our fragment shader information
    GLhandleARB m_hFragmentShader;

    /// This handle stores our program information which encompasses our shader
    GLhandleARB m_hProgramObject;

    GLint geometry_input_type;
    GLint geometry_output_type;
    GLint geometry_vertices_out;
};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif
