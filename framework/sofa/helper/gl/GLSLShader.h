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

#include <sofa/SofaFramework.h>

#ifndef SOFA_HAVE_GLEW
#error GL Shader support requires GLEW. Please define SOFA_HAVE_GLEW to use shaders.
#endif

#ifdef SOFA_HAVE_GLEW

#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>

#include <string>
#include <string.h>

#include <sofa/SofaFramework.h>
#include <vector>
#include <map>
#include <iostream>

namespace sofa
{

namespace helper
{

namespace gl
{



class SOFA_HELPER_API GLSLShader
{
public:

    GLSLShader();
    ~GLSLShader();

    /// This builds a header before any shader contents
    void AddHeader(const std::string& header);

    void AddDefineMacro(const std::string &name, const std::string &value);

    void SetShaderFileName(GLint target, const std::string& fileName);
    void SetVertexShaderFileName(const std::string& fileName)   { SetShaderFileName(GL_VERTEX_SHADER_ARB, fileName); }
    void SetFragmentShaderFileName(const std::string& fileName) { SetShaderFileName(GL_FRAGMENT_SHADER_ARB, fileName); }
#ifdef GL_GEOMETRY_SHADER_EXT
    void SetGeometryShaderFileName(const std::string& fileName) { SetShaderFileName(GL_GEOMETRY_SHADER_EXT, fileName); }
#endif
#ifdef GL_TESS_CONTROL_SHADER
    void SetTessellationControlShaderFileName(const std::string& fileName) { SetShaderFileName(GL_TESS_CONTROL_SHADER, fileName); }
#endif
#ifdef GL_TESS_EVALUATION_SHADER
    void SetTessellationEvaluationShaderFileName(const std::string& fileName) { SetShaderFileName(GL_TESS_EVALUATION_SHADER, fileName); }
#endif

    std::string GetShaderStageName(GLint target);


    /// This loads our text file for each shader and returns it in a string
    std::string LoadTextFile(const std::string& strFile);

    /// This is used to load all of the extensions and checks compatibility.
    static bool InitGLSL();

    static GLhandleARB GetActiveShaderProgram();
    static void  SetActiveShaderProgram(GLhandleARB s);

    // This loads all shaders previously set with Set*ShaderFileName() methods
    void InitShaders();

#ifdef GL_GEOMETRY_SHADER_EXT
    /// This loads a vertex, geometry and fragment shader
    void InitShaders(const std::string& strVertex, const std::string& strGeometry, const std::string& strFragment)
    {
        SetVertexShaderFileName(strVertex);
        SetGeometryShaderFileName(strGeometry);
        SetFragmentShaderFileName(strFragment);
        InitShaders();
    }
#endif

    /// This loads a vertex and fragment shader
    void InitShaders(const std::string& strVertex, const std::string& strFragment)
    {
        SetVertexShaderFileName(strVertex);
        SetFragmentShaderFileName(strFragment);
        InitShaders();
    }

    /// This returns an ID for a variable in our shader
    GLint GetVariable(std::string strVariable);

    /// This returns an ID for an attribute variable in our shader
    GLint GetAttributeVariable(std::string strVariable);

    /// These are our basic get functions for our private data
    /// @{
    bool        IsReady() const { return m_hProgramObject != 0; }
    GLhandleARB GetProgram() const	{	return m_hProgramObject; }
    std::string GetShaderFileName(GLint type) const;
    GLhandleARB GetShaderID(GLint type) const; //	{	std::map<GLint,GLhandleARB>::const_iterator it = m_hShaders.find(type); return (it.second ? *it.first : 0); }
    std::string GetVertexShaderFileName  () const { return GetShaderFileName(GL_VERTEX_SHADER_ARB); }
    GLhandleARB GetVertexShaderID        () const { return GetShaderID      (GL_VERTEX_SHADER_ARB); }
    std::string GetFragmentShaderFileName() const { return GetShaderFileName(GL_FRAGMENT_SHADER_ARB); }
    GLhandleARB GetFragmentShaderID      () const { return GetShaderID      (GL_FRAGMENT_SHADER_ARB); }
#ifdef GL_GEOMETRY_SHADER_EXT
    std::string GetGeometryShaderFileName() const { return GetShaderFileName(GL_VERTEX_SHADER_ARB); }
    GLhandleARB GetGeometryShaderID      () const { return GetShaderID      (GL_VERTEX_SHADER_ARB); }
#endif
#ifdef GL_TESS_CONTROL_SHADER
    std::string GetTessellationControlShaderFileName() { return GetShaderFileName(GL_TESS_CONTROL_SHADER); }
    GLhandleARB GetTessellationControlShaderID      () { return GetShaderID      (GL_TESS_CONTROL_SHADER); }
#endif
#ifdef GL_TESS_EVALUATION_SHADER
    std::string GetTessellationEvaluationShaderFileName() { return GetShaderFileName(GL_TESS_EVALUATION_SHADER); }
    GLhandleARB GetTessellationEvaluationShaderID      () { return GetShaderID      (GL_TESS_EVALUATION_SHADER); }
#endif
    /// @}

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

#ifdef GL_GEOMETRY_SHADER_EXT
    GLint GetGeometryInputType() { return geometry_input_type; }
    void  SetGeometryInputType(GLint v) { geometry_input_type = v; }

    GLint GetGeometryOutputType() { return geometry_output_type; }
    void  SetGeometryOutputType(GLint v) { geometry_output_type = v; }

    GLint GetGeometryVerticesOut() { return geometry_vertices_out; }
    void  SetGeometryVerticesOut(GLint v) { geometry_vertices_out = v; }
#endif

protected:

    bool CompileShader(GLint target, const std::string& fileName, const std::string& header);

    std::string header;

    std::map<GLint, std::string> m_hFileNames;
    std::map<GLint, GLhandleARB> m_hShaders;

    /// This handle stores our program information which encompasses our shader
    GLhandleARB m_hProgramObject;

#ifdef GL_GEOMETRY_SHADER_EXT
    GLint geometry_input_type;
    GLint geometry_output_type;
    GLint geometry_vertices_out;
#endif

};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif /* SOFA_HAVE_GLEW */

#endif /* SOFA_HELPER_GL_GLSLSHADER_H */
