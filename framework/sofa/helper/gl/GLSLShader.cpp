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
#include <sofa/helper/gl/GLSLShader.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>

namespace sofa
{

namespace helper
{

namespace gl
{

bool GLSLShader::InitGLSL()
{
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
    // Return a success!
    return true;
}

GLSLShader::GLSLShader()
{
    m_hVertexShader = 0; //NULL;
    m_hGeometryShader = 0; //NULL;
    m_hFragmentShader = 0; //NULL;
    m_hProgramObject = 0; //NULL;
    geometry_input_type = -1;
    geometry_output_type = -1;
    geometry_vertices_out = -1;
    header = "";
}

GLSLShader::~GLSLShader()
{
    // BUGFIX: if the GL context is gone, this can crash the application on exit -- Jeremie A.
    //Release();
}

void GLSLShader::AddHeader(const std::string &header)
{
    this->header += header;
    this->header += "\n";
}

void GLSLShader::AddDefineMacro(const std::string &name, const std::string &value)
{
    AddHeader("#define " + name + " " + value);
}

///	This function loads and returns a text file for our shaders
std::string GLSLShader::LoadTextFile(const std::string& strFile)
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

std::string CombineHeaders(std::string header, const std::string &shaderStage, std::string source)
{
    std::size_t spos = 1;
    // Skip #version
    if (source.size() > spos + 8 && (source.substr(spos,8)).compare(std::string("#version")) == 0)
    {
        spos = source.find('\n', spos+8);
        spos++;
    }


    // Skip #extension strings
    while (spos != std::string::npos && source.size() > spos + 10 && source.substr(spos,10) == std::string("#extension"))
    {
        spos = source.find('\n', spos+10);
        spos++;
    }

    header += "#define " + shaderStage + '\n';

    if (spos == 0)
        source = header + source;
    else if (spos == std::string::npos)
        source = source + "\n" + header;
    else
        source = source.substr(0, spos) + header + source.substr(spos);
    return source;
}

///	This function compiles a shader and check the log
bool GLSLShader::CompileShader(GLint target, const std::string& fileName, const std::string& header, GLhandleARB& shader)
{
    std::string source = LoadTextFile(fileName);
    std::string shaderStage;

    const char* stype = "";
    if (target == GL_VERTEX_SHADER_ARB)
    {
        stype = "vertex";
        shaderStage = "VertexShader";
    }
    else if (target == GL_FRAGMENT_SHADER_ARB)
    {
        stype = "fragment";
        shaderStage = "FragmentShader";
    }
#ifdef GL_GEOMETRY_SHADER_EXT
    else if (target == GL_GEOMETRY_SHADER_EXT)
    {
        stype = "geometry";
        shaderStage = "GeometryShader";
    }
#endif
    else
    {
        std::cerr << "Unknown shader target, refusing to load the shader..." << std::endl;
        return false;
    }

    source = CombineHeaders(header, shaderStage, source);

    shader = glCreateShaderObjectARB(target);

    const char* src = source.c_str();

    glShaderSourceARB(shader, 1, &src, NULL);

    glCompileShaderARB(shader);

    GLint compiled = 0, length = 0, laux = 0;
    glGetObjectParameterivARB(shader, GL_OBJECT_COMPILE_STATUS_ARB, &compiled);
    if (!compiled) std::cerr << "ERROR: Compilation of "<<stype<<" shader failed:\n"<<src<<std::endl;
    //     else std::cout << "Compilation of "<<stype<<" shader OK" << std::endl;
    glGetObjectParameterivARB(shader, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
    if (length > 1)
    {
        std::cerr << "File: " << fileName << std::endl;
        if (!header.empty()) std::cerr << "Header:\n" << header << std::endl;
        GLcharARB *logString = (GLcharARB *)malloc((length+1) * sizeof(GLcharARB));
        glGetInfoLogARB(shader, length, &laux, logString);
        std::cerr << logString << std::endl;
        free(logString);
    }
    return (compiled!=0);
}

///	This function loads a vertex and fragment shader file
void GLSLShader::InitShaders(const std::string& strVertex, const std::string& strGeometry, const std::string& strFragment)
{
    // Make sure the user passed in at least a vertex and fragment shader file
    if(!strVertex.length() || !strFragment.length())
        return;

    // If any of our shader pointers are set, let's free them first.
    if(m_hVertexShader || m_hGeometryShader || m_hFragmentShader || m_hProgramObject)
        Release();

    bool ready = true;

    // Now we load and compile the shaders from their respective files
    ready &= CompileShader( GL_VERTEX_SHADER_ARB, strVertex, header, m_hVertexShader );
    if (!strGeometry.empty())
    {
#ifdef GL_GEOMETRY_SHADER_EXT
        ready &= CompileShader( GL_GEOMETRY_SHADER_EXT, strGeometry, header, m_hGeometryShader );
#else
        std::cerr << "SHADER ERROR: GL_GEOMETRY_SHADER_EXT not defined. Please use a recent version of GLEW.\n";
        ready = false;
#endif
    }
    ready &= CompileShader( GL_FRAGMENT_SHADER_ARB, strFragment, header, m_hFragmentShader );

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
//     else std::cout << "Link of program shader OK" << std::endl;
    glGetObjectParameterivARB(m_hProgramObject, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
    if (length > 1)
    {
        GLcharARB *logString = (GLcharARB *)malloc((length+1) * sizeof(GLcharARB));
        glGetInfoLogARB(m_hProgramObject, length, &laux, logString);
        std::cerr << logString << std::endl;
        free(logString);
        std::cerr << "Vertex shader file: " << strVertex << std::endl;
        if (!strGeometry.empty())
            std::cerr << "Geometry shader file: " << strGeometry << std::endl;
        std::cerr << "Fragment shader file: " << strFragment << std::endl;
    }

    // Now, let's turn off the shader initially.
    glUseProgramObjectARB(0);
}


void GLSLShader::SetInt(GLint variable, int newValue)                              { if (variable!=-1) glUniform1iARB(variable, newValue);       }
void GLSLShader::SetInt2(GLint variable, int i1, int i2)                           { if (variable!=-1) glUniform2iARB(variable, i1, i2);         }
void GLSLShader::SetInt3(GLint variable, int i1, int i2, int i3)                   { if (variable!=-1) glUniform3iARB(variable, i1, i2, i3);     }
void GLSLShader::SetInt4(GLint variable, int i1, int i2, int i3, int i4)           { if (variable!=-1) glUniform4iARB(variable, i1, i2, i3, i4); }
void GLSLShader::SetFloat(GLint variable, float newValue)                          { if (variable!=-1) glUniform1fARB(variable, newValue);       }
void GLSLShader::SetFloat2(GLint variable, float v0, float v1)                     { if (variable!=-1) glUniform2fARB(variable, v0, v1);         }
void GLSLShader::SetFloat3(GLint variable, float v0, float v1, float v2)           { if (variable!=-1) glUniform3fARB(variable, v0, v1, v2);     }
void GLSLShader::SetFloat4(GLint variable, float v0, float v1, float v2, float v3) { if (variable!=-1) glUniform4fARB(variable, v0, v1, v2, v3); }

void GLSLShader::SetIntVector(GLint variable, GLsizei count, const GLint *value)     { if (variable!=-1) glUniform1ivARB(variable, count, value);   }
void GLSLShader::SetIntVector2(GLint variable, GLsizei count, const GLint *value)    { if (variable!=-1) glUniform2ivARB(variable, count, value);   }
void GLSLShader::SetIntVector3(GLint variable, GLsizei count, const GLint *value)    { if (variable!=-1) glUniform3ivARB(variable, count, value);   }
void GLSLShader::SetIntVector4(GLint variable, GLsizei count, const GLint *value)    { if (variable!=-1) glUniform4ivARB(variable, count, value);   }

void GLSLShader::SetFloatVector(GLint variable, GLsizei count, const float *value) { if (variable!=-1) glUniform1fvARB(variable, count, value);   }
void GLSLShader::SetFloatVector2(GLint variable, GLsizei count, const float *value) { if (variable!=-1) glUniform2fvARB(variable, count, value);   }
void GLSLShader::SetFloatVector3(GLint variable, GLsizei count, const float *value) { if (variable!=-1) glUniform3fvARB(variable, count, value);   }
void GLSLShader::SetFloatVector4(GLint variable, GLsizei count, const float *value) { if (variable!=-1) glUniform4fvARB(variable, count, value);   }

void GLSLShader::SetMatrix2(GLint variable, GLsizei count, GLboolean transpose, const GLfloat *value) { if (variable!=-1) glUniformMatrix2fv(variable, count, transpose, value);   }
void GLSLShader::SetMatrix3(GLint variable, GLsizei count, GLboolean transpose, const GLfloat *value) { if (variable!=-1) glUniformMatrix3fv(variable, count, transpose, value);   }
void GLSLShader::SetMatrix4(GLint variable, GLsizei count, GLboolean transpose, const GLfloat *value) { if (variable!=-1) glUniformMatrix4fv(variable, count, transpose, value);   }

#ifdef GL_VERSION_2_1
void GLSLShader::SetMatrix2x3(GLint variable,GLsizei count,GLboolean transpose, const GLfloat *value) { if (variable!=-1) glUniformMatrix2x3fv(variable, count, transpose, value);   }
void GLSLShader::SetMatrix3x2(GLint variable,GLsizei count,GLboolean transpose, const GLfloat *value) { if (variable!=-1) glUniformMatrix3x2fv(variable, count, transpose, value);   }
void GLSLShader::SetMatrix2x4(GLint variable,GLsizei count,GLboolean transpose, const GLfloat *value) { if (variable!=-1) glUniformMatrix2x4fv(variable, count, transpose, value);   }
void GLSLShader::SetMatrix4x2(GLint variable,GLsizei count,GLboolean transpose, const GLfloat *value) { if (variable!=-1) glUniformMatrix4x2fv(variable, count, transpose, value);   }
void GLSLShader::SetMatrix3x4(GLint variable,GLsizei count,GLboolean transpose, const GLfloat *value) { if (variable!=-1) glUniformMatrix3x4fv(variable, count, transpose, value);   }
void GLSLShader::SetMatrix4x3(GLint variable,GLsizei count,GLboolean transpose, const GLfloat *value) { if (variable!=-1) glUniformMatrix4x3fv(variable, count, transpose, value);   }
#endif

// These 2 functions turn on and off our shader
void GLSLShader::TurnOn()	{ if (m_hProgramObject) glUseProgramObjectARB(m_hProgramObject); }
void GLSLShader::TurnOff()	{ if (m_hProgramObject) glUseProgramObjectARB(0);                }

///	This function returns a variable ID for a shader variable
GLint GLSLShader::GetVariable(std::string strVariable)
{
    // If we don't have an active program object, let's return -1
    if(!m_hProgramObject)
        return -1;

    // This returns the variable ID for a variable that is used to find
    // the address of that variable in memory.
    return glGetUniformLocationARB(m_hProgramObject, strVariable.c_str());
}

GLint GLSLShader::GetAttributeVariable(std::string strVariable)
{
    // If we don't have an active program object, let's return -1
    if(!m_hProgramObject)
        return -1;

    // This returns the variable ID for a variable that is used to find
    // the address of that variable in memory.
    return glGetAttribLocationARB(m_hProgramObject, strVariable.c_str());
}


///	This function frees all of our shader data
void GLSLShader::Release()
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
