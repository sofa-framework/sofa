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
//*******************************************************************//
//    OpenGL Shader class                                            //
//                                                                   //
//    Based on code from Ben Humphrey    / digiben@gametutorilas.com //
//                                                                   //
//*******************************************************************//

#include <sofa/helper/gl/GLshader.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>

#ifndef SOFA_HAVE_GLEW
#if !defined(_WIN32) && !defined(__APPLE__)
#  include <GL/glx.h>
#endif
#ifdef __APPLE__
//CHANGE(Jeremie A.): NS* methods to access symbols are deprecated in favor of standard dl* methods
//#include "mach-o/dyld.h"
#include <dlfcn.h>
#endif
#endif

namespace sofa
{

namespace helper
{

namespace gl
{

#ifndef SOFA_HAVE_GLEW

// The function pointers for shaders
PFNGLCREATESHADEROBJECTARBPROC	glCreateShaderObjectARB = NULL;
PFNGLSHADERSOURCEARBPROC		glShaderSourceARB = NULL;
PFNGLCOMPILESHADERARBPROC		glCompileShaderARB = NULL;
PFNGLCREATEPROGRAMOBJECTARBPROC glCreateProgramObjectARB = NULL;
PFNGLATTACHOBJECTARBPROC		glAttachObjectARB = NULL;
PFNGLLINKPROGRAMARBPROC			glLinkProgramARB = NULL;
PFNGLUSEPROGRAMOBJECTARBPROC	glUseProgramObjectARB = NULL;
PFNGLUNIFORM1IARBPROC			glUniform1iARB = NULL;
PFNGLUNIFORM1FARBPROC			glUniform1fARB = NULL;
PFNGLUNIFORM2FARBPROC			glUniform2fARB = NULL;
PFNGLUNIFORM3FARBPROC			glUniform3fARB = NULL;
PFNGLUNIFORM4FARBPROC			glUniform4fARB = NULL;
PFNGLGETUNIFORMLOCATIONARBPROC	glGetUniformLocationARB = NULL;
PFNGLDETACHOBJECTARBPROC		glDetachObjectARB = NULL;
PFNGLDELETEOBJECTARBPROC		glDeleteObjectARB = NULL;
PFNGLACTIVETEXTUREARBPROC		glActiveTextureARB = NULL;
PFNGLMULTITEXCOORD2FARBPROC		glMultiTexCoord2fARB = NULL;
PFNGLGETOBJECTPARAMETERIV		glGetObjectParameterivARB = NULL;
PFNGLGETINFOLOGARBPROC			glGetInfoLogARB = NULL;


///////////////////////////////////// INIT GLSL \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////	This function initializes all of our GLSL functions and checks availability.
/////
///////////////////////////////////// INIT GLSL \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

#ifdef __APPLE__
//CHANGE(Jeremie A.): NS* methods to access symbols are deprecated in favor of standard dl* methods
#if 0
void *NSGLGetProcAddress(const char *name)
{
    NSSymbol symbol;

    /* Prepend a '_' for the Unix C symbol mangling convention */
    char* symbolName = (char*)malloc(strlen(name) + 2);
    if (!symbolName)
    {
        fprintf(stderr, "Failed to allocate memory for NSGLGetProcAddress\n");
        return NULL;
    }
    symbolName[0] = '_';
    strcpy(symbolName + 1, name);

    if (!NSIsSymbolNameDefined(symbolName))
    {
        free(symbolName);
        return NULL;
    }

    symbol = NSLookupAndBindSymbol(symbolName);
    free(symbolName);
    if (!symbol)
    {
        return NULL;
    }

    return NSAddressOfSymbol(symbol);
}
#endif
#endif

#if defined (WIN32)
PROC glewGetProcAddress(const char* name)
#elif defined(__APPLE__)
void (*glewGetProcAddress(const char* name))
#else
void (*glewGetProcAddress(const char* name))(void)
#endif
{
#if defined(_WIN32)
    return wglGetProcAddress((LPCSTR)name);
#elif defined(__APPLE__)
//CHANGE(Jeremie A.): NS* methods to access symbols are deprecated in favor of standard dl* methods
//    return NSGLGetProcAddress(name);
    return dlsym(RTLD_DEFAULT, name);
#elif defined(__sgi) || defined(__sun)
    return dlGetProcAddress(name);
#else /* __linux */
    return (*glXGetProcAddressARB)((const GLubyte*)name);
#endif
}

#endif

bool CShader::InitGLSL()
{
#ifdef SOFA_HAVE_GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
        return false;
    }
    fprintf(stdout, "CShader: Using GLEW %s\n", glewGetString(GLEW_VERSION));

    // Make sure find the GL_ARB_shader_objects extension so we can use shaders.
    if(!GLEW_ARB_shader_objects)
    {
        fprintf(stderr, "Error: GL_ARB_shader_objects extension not supported!\n");
        return false;
    }

    // Make sure we support the GLSL shading language 1.0
    if(!GLEW_ARB_shading_language_100)
    {
        fprintf(stderr, "Error: GL_ARB_shading_language_100 extension not supported!\n");
        return false;
    }


#else
    // This grabs a list of all the video card's extensions it supports
    char *szGLExtensions = (char*)glGetString(GL_EXTENSIONS);

    // Make sure find the GL_ARB_shader_objects extension so we can use shaders.
    if(!strstr(szGLExtensions, "GL_ARB_shader_objects"))
    {
        fprintf(stderr, "Error: GL_ARB_shader_objects extension not supported!\n");
        return false;
    }

    // Make sure we support the GLSL shading language 1.0
    if(!strstr(szGLExtensions, "GL_ARB_shading_language_100"))
    {
        fprintf(stderr, "Error: GL_ARB_shading_language_100 extension not supported!\n");
        return false;
    }

    // Now let's set all of our function pointers for our extension functions
    glCreateShaderObjectARB = (PFNGLCREATESHADEROBJECTARBPROC)glewGetProcAddress("glCreateShaderObjectARB");
    glShaderSourceARB = (PFNGLSHADERSOURCEARBPROC)glewGetProcAddress("glShaderSourceARB");
    glCompileShaderARB = (PFNGLCOMPILESHADERARBPROC)glewGetProcAddress("glCompileShaderARB");
    glCreateProgramObjectARB = (PFNGLCREATEPROGRAMOBJECTARBPROC)glewGetProcAddress("glCreateProgramObjectARB");
    glAttachObjectARB = (PFNGLATTACHOBJECTARBPROC)glewGetProcAddress("glAttachObjectARB");
    glLinkProgramARB = (PFNGLLINKPROGRAMARBPROC)glewGetProcAddress("glLinkProgramARB");
    glUseProgramObjectARB = (PFNGLUSEPROGRAMOBJECTARBPROC)glewGetProcAddress("glUseProgramObjectARB");
    glUniform1iARB = (PFNGLUNIFORM1IARBPROC)glewGetProcAddress("glUniform1iARB");
    glUniform1fARB = (PFNGLUNIFORM1FARBPROC)glewGetProcAddress("glUniform1fARB");
    glUniform2fARB = (PFNGLUNIFORM2FARBPROC)glewGetProcAddress("glUniform2fARB");
    glUniform3fARB = (PFNGLUNIFORM3FARBPROC)glewGetProcAddress("glUniform3fARB");
    glUniform4fARB = (PFNGLUNIFORM4FARBPROC)glewGetProcAddress("glUniform4fARB");
    glGetUniformLocationARB = (PFNGLGETUNIFORMLOCATIONARBPROC)glewGetProcAddress("glGetUniformLocationARB");
    glDetachObjectARB = (PFNGLDETACHOBJECTARBPROC)glewGetProcAddress("glDetachObjectARB");
    glDeleteObjectARB  = (PFNGLDELETEOBJECTARBPROC)glewGetProcAddress("glDeleteObjectARB");
    glActiveTextureARB = (PFNGLACTIVETEXTUREARBPROC)glewGetProcAddress("glActiveTextureARB");
    glMultiTexCoord2fARB = (PFNGLMULTITEXCOORD2FARBPROC)glewGetProcAddress("glMultiTexCoord2fARB");
    glGetObjectParameterivARB = (PFNGLGETOBJECTPARAMETERIV)glewGetProcAddress("glGetObjectParameterivARB");
    glGetInfoLogARB = (PFNGLGETINFOLOGARBPROC)glewGetProcAddress("glGetInfoLogARB");
#endif
    // Return a success!
    return true;
}

CShader::CShader()
{
    m_hVertexShader = 0; //NULL;
    m_hGeometryShader = 0; //NULL;
    m_hFragmentShader = 0; //NULL;
    m_hProgramObject = 0; //NULL;
    geometry_input_type = -1;
    geometry_output_type = -1;
    geometry_vertices_out = -1;
}

CShader::~CShader()
{
    // BUGFIX: if the GL context is gone, this can crash the application on exit -- Jeremie A.
    //Release();
}

///////////////////////////////// LOAD TEXT FILE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////	This function loads and returns a text file for our shaders
/////
///////////////////////////////// LOAD TEXT FILE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

std::string CShader::LoadTextFile(const std::string& strFile)
{
    // Open the file passed in
    std::ifstream fin(strFile.c_str());

    // Make sure we opened the file correctly
    if(!fin)
        return "";

    std::string strLine = "";
    std::string strText = "";

    // Go through and store each line in the text file within a "string" object
    while(std::getline(fin, strLine))
    {
        strText = strText + "\n" + strLine;
    }

    // Close our file
    fin.close();

    // Return the text file's data
    return strText;
}

///////////////////////////////// INIT SHADERS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////	This function compiles a shader and check the log
/////
///////////////////////////////// INIT SHADERS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

bool CShader::CompileShader(GLint target, const std::string& source, GLhandleARB& shader)
{
    const char* stype = "";
    if (target == GL_VERTEX_SHADER_ARB) stype = "vertex";
    else if (target == GL_FRAGMENT_SHADER_ARB) stype = "fragment";
#ifdef GL_GEOMETRY_SHADER_EXT
    else if (target == GL_GEOMETRY_SHADER_EXT) stype = "geometry";
#endif

    shader = glCreateShaderObjectARB(target);

    const char* src = source.c_str();

    glShaderSourceARB(shader, 1, &src, NULL);

    glCompileShaderARB(shader);

    GLint compiled = 0, length = 0, laux = 0;
    glGetObjectParameterivARB(shader, GL_OBJECT_COMPILE_STATUS_ARB, &compiled);
    if (!compiled) std::cerr << "ERROR: Compilation of "<<stype<<" shader failed:\n"<<src<<std::endl;
    else std::cout << "Compilation of "<<stype<<" shader OK" << std::endl;
    glGetObjectParameterivARB(shader, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
    if (length)
    {
        GLcharARB *logString = (GLcharARB *)malloc((length+1) * sizeof(GLcharARB));
        glGetInfoLogARB(shader, length, &laux, logString);
        std::cerr << logString << std::endl;
        free(logString);
    }
    return (compiled!=0);
}

///////////////////////////////// INIT SHADERS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////	This function loads a vertex and fragment shader file
/////
///////////////////////////////// INIT SHADERS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

void CShader::InitShaders(const std::string& strVertex, const std::string& strGeometry, const std::string& strFragment)
{
    // Make sure the user passed in at least a vertex and fragment shader file
    if(!strVertex.length() || !strFragment.length())
        return;

    // If any of our shader pointers are set, let's free them first.
    if(m_hVertexShader || m_hGeometryShader || m_hFragmentShader || m_hProgramObject)
        Release();

    bool ready = true;

    // Now we load and compile the shaders from their respective files
    ready &= CompileShader( GL_VERTEX_SHADER_ARB, LoadTextFile(strVertex), m_hVertexShader );
    if (!strGeometry.empty())
    {
#ifdef GL_GEOMETRY_SHADER_EXT
        ready &= CompileShader( GL_GEOMETRY_SHADER_EXT, LoadTextFile(strGeometry), m_hGeometryShader );
#else
        std::cerr << "SHADER ERROR: GL_GEOMETRY_SHADER_EXT not defined. Please use a recent version of GLEW.\n";
        ready = false;
#endif
    }
    ready &= CompileShader( GL_FRAGMENT_SHADER_ARB, LoadTextFile(strFragment), m_hFragmentShader );

    if (!ready)
    {
        std::cerr << "SHADER compilation failed.\n";
        return;
    }

    // Next we create a program object to represent our shaders
    m_hProgramObject = glCreateProgramObjectARB();

    // We attach each shader we just loaded to our program object
    glAttachObjectARB(m_hProgramObject, m_hVertexShader);
#ifdef GL_GEOMETRY_SHADER_EXT
    if (m_hGeometryShader)
        glAttachObjectARB(m_hProgramObject, m_hGeometryShader);
#endif
    glAttachObjectARB(m_hProgramObject, m_hFragmentShader);

#ifdef GL_GEOMETRY_SHADER_EXT
    if (m_hGeometryShader)
    {
        if (geometry_input_type != -1) glProgramParameteriEXT(m_hProgramObject, GL_GEOMETRY_INPUT_TYPE_EXT, geometry_input_type );
        if (geometry_output_type != -1) glProgramParameteriEXT(m_hProgramObject, GL_GEOMETRY_OUTPUT_TYPE_EXT, geometry_output_type );
        if (geometry_vertices_out != -1) glProgramParameteriEXT(m_hProgramObject, GL_GEOMETRY_VERTICES_OUT_EXT, geometry_vertices_out );
    }
#endif
    // Our last init function is to link our program object with OpenGL
    glLinkProgramARB(m_hProgramObject);

    GLint linked = 0, length = 0, laux = 0;
    glGetObjectParameterivARB(m_hProgramObject, GL_OBJECT_LINK_STATUS_ARB, &linked);
    if (!linked) std::cerr << "ERROR: Link of program shader failed:\n"<<std::endl;
    else std::cout << "Link of program shader OK" << std::endl;
    glGetObjectParameterivARB(m_hProgramObject, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
    if (length)
    {
        GLcharARB *logString = (GLcharARB *)malloc((length+1) * sizeof(GLcharARB));
        glGetInfoLogARB(m_hProgramObject, length, &laux, logString);
        std::cerr << logString << std::endl;
        free(logString);
    }

    // Now, let's turn off the shader initially.
    glUseProgramObjectARB(0);
}


void CShader::SetInt(GLint variable, int newValue)                              { if (variable!=-1) glUniform1iARB(variable, newValue);       }
void CShader::SetInt2(GLint variable, int i1, int i2)                           { if (variable!=-1) glUniform2iARB(variable, i1, i2);         }
void CShader::SetInt3(GLint variable, int i1, int i2, int i3)                   { if (variable!=-1) glUniform3iARB(variable, i1, i2, i3);     }
void CShader::SetInt4(GLint variable, int i1, int i2, int i3, int i4)           { if (variable!=-1) glUniform4iARB(variable, i1, i2, i3, i4); }
void CShader::SetFloat(GLint variable, float newValue)                          { if (variable!=-1) glUniform1fARB(variable, newValue);       }
void CShader::SetFloat2(GLint variable, float v0, float v1)                     { if (variable!=-1) glUniform2fARB(variable, v0, v1);         }
void CShader::SetFloat3(GLint variable, float v0, float v1, float v2)           { if (variable!=-1) glUniform3fARB(variable, v0, v1, v2);     }
void CShader::SetFloat4(GLint variable, float v0, float v1, float v2, float v3) { if (variable!=-1) glUniform4fARB(variable, v0, v1, v2, v3); }

void CShader::SetIntVector(GLint variable, GLsizei count, const GLint *value)     { if (variable!=-1) glUniform1ivARB(variable, count, value);   }
void CShader::SetIntVector2(GLint variable, GLsizei count, const GLint *value)    { if (variable!=-1) glUniform2ivARB(variable, count, value);   }
void CShader::SetIntVector3(GLint variable, GLsizei count, const GLint *value)    { if (variable!=-1) glUniform3ivARB(variable, count, value);   }
void CShader::SetIntVector4(GLint variable, GLsizei count, const GLint *value)    { if (variable!=-1) glUniform4ivARB(variable, count, value);   }

void CShader::SetFloatVector(GLint variable, GLsizei count, const float *value) { if (variable!=-1) glUniform1fvARB(variable, count, value);   }
void CShader::SetFloatVector2(GLint variable, GLsizei count, const float *value) { if (variable!=-1) glUniform2fvARB(variable, count, value);   }
void CShader::SetFloatVector3(GLint variable, GLsizei count, const float *value) { if (variable!=-1) glUniform3fvARB(variable, count, value);   }
void CShader::SetFloatVector4(GLint variable, GLsizei count, const float *value) { if (variable!=-1) glUniform4fvARB(variable, count, value);   }


// These 2 functions turn on and off our shader
void CShader::TurnOn()	{ if (m_hProgramObject) glUseProgramObjectARB(m_hProgramObject); }
void CShader::TurnOff()	{ if (m_hProgramObject) glUseProgramObjectARB(0);                }

///////////////////////////////// GET VARIABLE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////	This function returns a variable ID for a shader variable
/////
///////////////////////////////// GET VARIABLE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

GLint CShader::GetVariable(std::string strVariable)
{
    // If we don't have an active program object, let's return -1
    if(!m_hProgramObject)
        return -1;

    // This returns the variable ID for a variable that is used to find
    // the address of that variable in memory.
    return glGetUniformLocationARB(m_hProgramObject, strVariable.c_str());
}


///////////////////////////////// RELEASE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////	This function frees all of our shader data
/////
///////////////////////////////// RELEASE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

void CShader::Release()
{
    // If our vertex shader pointer is valid, free it
    if(m_hVertexShader)
    {
        glDetachObjectARB(m_hProgramObject, m_hVertexShader);
        glDeleteObjectARB(m_hVertexShader);
        m_hVertexShader = 0; //NULL;
    }

    // If our geometry shader pointer is valid, free it
    if(m_hGeometryShader)
    {
        glDetachObjectARB(m_hProgramObject, m_hGeometryShader);
        glDeleteObjectARB(m_hGeometryShader);
        m_hGeometryShader = 0; //NULL;
    }

    // If our fragment shader pointer is valid, free it
    if(m_hFragmentShader)
    {
        glDetachObjectARB(m_hProgramObject, m_hFragmentShader);
        glDeleteObjectARB(m_hFragmentShader);
        m_hFragmentShader = 0; //NULL;
    }

    // If our program object pointer is valid, free it
    if(m_hProgramObject)
    {
        glDeleteObjectARB(m_hProgramObject);
        m_hProgramObject = 0; //NULL;
    }
}

} // namespace gl

} // namespace helper

} // namespace sofa

// (C) 2000-2005 GameTutorials
