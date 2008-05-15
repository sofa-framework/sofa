/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_HELPER_GL_GLSHADER_H
#define SOFA_HELPER_GL_GLSHADER_H

#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>

#include <string>
#include <string.h>


namespace sofa
{

namespace helper
{

namespace gl
{

#ifndef SOFA_HAVE_GLEW

#if defined (WIN32)
PROC glewGetProcAddress(const char* name);
#elif defined(__APPLE__)
void (*glewGetProcAddress(const char* name));
#else
void (*glewGetProcAddress(const char* name))(void);
#endif

// This is a define that we use for our function pointers
#ifdef APIENTRY
#define APIENTRYP APIENTRY *
#else
#define APIENTRYP *
#endif

// Here we include the vertex and fragment shader defines
#define GL_VERTEX_SHADER_ARB              0x8B31
#define GL_FRAGMENT_SHADER_ARB            0x8B30

#define GL_VERTEX_PROGRAM_ARB             0x8620
#define GL_FRAGMENT_PROGRAM_ARB           0x8804

#define GL_OBJECT_COMPILE_STATUS_ARB	  0x8B81
#define GL_OBJECT_LINK_STATUS_ARB		  0x8B82
#define GL_OBJECT_INFO_LOG_LENGTH_ARB	  0x8B84
#define GL_OBJECT_DELETE_STATUS_ARB		  0x8B80

// These are for our multi-texture defines
#define GL_TEXTURE0_ARB                   0x84C0
#define GL_TEXTURE1_ARB                   0x84C1
#define GL_TEXTURE2_ARB                   0x84C2
#define GL_TEXTURE3_ARB                   0x84C3
#define GL_TEXTURE4_ARB                   0x84C4

#define GL_DEPTH_TEXTURE_MODE_ARB         0x884B
#define GL_TEXTURE_COMPARE_MODE_ARB       0x884C
#define GL_TEXTURE_COMPARE_FUNC_ARB       0x884D
#define GL_COMPARE_R_TO_TEXTURE_ARB       0x884E

// This is what GL uses for handles when using extensions
typedef unsigned int GLhandleARB;
typedef char GLcharARB;

// Below are all of our function pointer typedefs for all the extensions we need
typedef GLhandleARB (APIENTRYP PFNGLCREATESHADEROBJECTARBPROC) (GLenum shaderType);
typedef void (APIENTRYP PFNGLSHADERSOURCEARBPROC) (GLhandleARB shaderObj, GLsizei count, const GLcharARB* *string, const GLint *length);
typedef void (APIENTRYP PFNGLCOMPILESHADERARBPROC) (GLhandleARB shaderObj);
typedef GLhandleARB (APIENTRYP PFNGLCREATEPROGRAMOBJECTARBPROC) (void);
typedef void (APIENTRYP PFNGLATTACHOBJECTARBPROC) (GLhandleARB containerObj, GLhandleARB obj);
typedef void (APIENTRYP PFNGLLINKPROGRAMARBPROC) (GLhandleARB programObj);
typedef void (APIENTRYP PFNGLUSEPROGRAMOBJECTARBPROC) (GLhandleARB programObj);
typedef void (APIENTRYP PFNGLUNIFORM1IARBPROC) (GLint location, GLint v0);
typedef void (APIENTRYP PFNGLUNIFORM1FARBPROC) (GLint location, GLfloat v0);
typedef void (APIENTRYP PFNGLUNIFORM2FARBPROC) (GLint location, GLfloat v0, GLfloat v1);
typedef void (APIENTRYP PFNGLUNIFORM3FARBPROC) (GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
typedef void (APIENTRYP PFNGLUNIFORM4FARBPROC) (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
typedef GLint (APIENTRYP PFNGLGETUNIFORMLOCATIONARBPROC) (GLhandleARB programObj, const GLcharARB *name);
typedef void (APIENTRYP PFNGLDETACHOBJECTARBPROC) (GLhandleARB containerObj, GLhandleARB attachedObj);
typedef void (APIENTRYP PFNGLDELETEOBJECTARBPROC) (GLhandleARB obj);
typedef void (APIENTRYP PFNGLPROGRAMLOCALPARAMETER4FARBPROC) (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
typedef void (APIENTRYP PFNGLBINDPROGRAMARBPROC) (GLenum target, GLuint program);
typedef void (APIENTRYP PFNGLGETOBJECTPARAMETERIV) (GLhandleARB obj, GLenum pname, GLint *params);
typedef void (APIENTRYP PFNGLGETINFOLOGARBPROC) (GLhandleARB obj, GLsizei maxLength, GLsizei* length, GLcharARB *infoLog);
typedef void (APIENTRYP PFNGLMULTITEXCOORD2FARBPROC) (GLenum target, GLfloat s, GLfloat t);
typedef void (APIENTRYP PFNGLACTIVETEXTUREARBPROC) (GLenum target);

/*
// Here we extern our functions to be used elsewhere
extern PFNGLCREATESHADEROBJECTARBPROC glCreateShaderObjectARB;
extern PFNGLSHADERSOURCEARBPROC glShaderSourceARB;
extern PFNGLCOMPILESHADERARBPROC glCompileShaderARB;
extern PFNGLCREATEPROGRAMOBJECTARBPROC glCreateProgramObjectARB;
extern PFNGLATTACHOBJECTARBPROC glAttachObjectARB;
extern PFNGLLINKPROGRAMARBPROC glLinkProgramARB;
extern PFNGLUSEPROGRAMOBJECTARBPROC glUseProgramObjectARB;
extern PFNGLUNIFORM1IARBPROC glUniform1iARB;
extern PFNGLUNIFORM1FARBPROC glUniform1fARB;
extern PFNGLUNIFORM2FARBPROC glUniform2fARB;
extern PFNGLUNIFORM3FARBPROC glUniform3fARB;
extern PFNGLUNIFORM4FARBPROC glUniform4fARB;
extern PFNGLGETUNIFORMLOCATIONARBPROC glGetUniformLocationARB;
extern PFNGLDETACHOBJECTARBPROC glDetachObjectARB;
extern PFNGLDELETEOBJECTARBPROC glDeleteObjectARB;
extern PFNGLPROGRAMLOCALPARAMETER4FARBPROC glProgramLocalParameter4fARB;
extern PFNGLBINDPROGRAMARBPROC glBindProgramARB;
extern PFNGLGETOBJECTPARAMETERIV glGetObjectParameterivARB;
extern PFNGLGETINFOLOGARBPROC glGetInfoLogARB;
*/
extern PFNGLMULTITEXCOORD2FARBPROC glMultiTexCoord2fARB;
extern PFNGLACTIVETEXTUREARBPROC glActiveTextureARB;

#endif

class CShader
{
public:

    CShader();
    ~CShader();

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

    bool CompileShader(GLint target, const std::string& source, GLhandleARB& shader);

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
