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
#include <sofa/helper/gl/GLSLShader.h>

#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>

#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace helper
{

namespace gl
{

class GLSLFileListener : public FileEventListener
{
public:
    /// This attribute is not owning the pointer.
    GLSLShader* m_glslshader ;

    GLSLFileListener(GLSLShader* owner){
        m_glslshader = owner ;
    }

    /// Inherited from FileEventListener
    void fileHasChanged(const std::string& filename){
        /// We are recompiling & re-initializing all the shaders...
        /// If this become a bottleneck we can do finer grain updates to
        /// speed up the thing.
        m_glslshader->forceReloadShaderFromFile(filename);
        m_glslshader->InitShaders() ;
    }
};


bool GLSLIsSupported = false;

bool GLSLShader::InitGLSL()
{
#ifdef PS3
    return false;
#else
    // Make sure find the GL_ARB_shader_objects extension so we can use shaders.
    if( !CanUseGlExtension("GL_ARB_shading_language_100") )
    {
        msg_error("GLSLShader") << "GL_ARB_shader_objects extension not supported !" ;
        return false;
    }

    // Make sure we support the GLSL shading language 1.0
    if( !CanUseGlExtension("GL_ARB_shading_language_100") )
    {
        msg_error("GLSLShader") << "GL_ARB_shading_language_100 extension not supported! ";
        return false;
    }
    GLSLIsSupported = true;
    return true;
#endif
}

GLhandleARB GLSLShader::GetActiveShaderProgram()
{
    if (!GLSLIsSupported) return 0;
    return glGetHandleARB(GL_PROGRAM_OBJECT_ARB);
}

void  GLSLShader::SetActiveShaderProgram(GLhandleARB s)
{
    if (!GLSLIsSupported) return;
    glUseProgramObjectARB(s);
}


GLSLShader::GLSLShader()
{
    m_hProgramObject = 0;
#ifdef GL_GEOMETRY_SHADER_EXT
    geometry_input_type = -1;
    geometry_output_type = -1;
    geometry_vertices_out = -1;
#endif
    header = "";
    m_filelistener = std::shared_ptr<FileEventListener>(new GLSLFileListener(this)) ;
}

GLSLShader::~GLSLShader()
{
    // BUGFIX: if the GL context is gone, this can crash the application on exit -- Jeremie A.
    //Release();
    if(m_filelistener){
        FileMonitor::removeListener(m_filelistener.get());
    }
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
void GLSLShader::SetShaderFileName(GLint target, const std::string& filename)
{
    if (filename.empty())
    {
        if(m_filelistener && !m_hShaderContents[target].filename.empty())
            FileMonitor::removeFileListener(m_hShaderContents[target].filename, m_filelistener.get()) ;
        m_hShaderContents.erase(target);
    }else{
        ShaderContents sc;
        sc.filename = filename;
        sc.text = LoadTextFile(filename);
        m_hShaderContents[target] = sc;
        if(m_filelistener)
            FileMonitor::addFile(filename, m_filelistener.get()) ;
    }
}

void GLSLShader::forceReloadShaderFromFile(const std::string& filename)
{
    for(auto& fn : m_hShaderContents){
        if(fn.second.filename == filename)
        {
            fn.second.text = "" ;
        }
    }
}

void GLSLShader::SetShaderFromString(GLint target, const std::string& str)
{
    if (str.empty())
    {
        m_hShaderContents.erase(target);
    }
    else {
        ShaderContents sc;
        sc.filename = "";
        sc.text = str;
        m_hShaderContents[target] = sc;
    }
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
    std::size_t spos = source.size() ? 1 : 0;
    int srcline = 1;
    // Skip #version
    if (source.size() > spos + 8 && (source.substr(spos,8)).compare(std::string("#version")) == 0)
    {
        spos = source.find('\n', spos+8);
        spos++;
        ++srcline;
    }


    // Skip #extension strings
    while (spos != std::string::npos && source.size() > spos + 10 && source.substr(spos,10) == std::string("#extension"))
    {
        spos = source.find('\n', spos+10);
        spos++;
        ++srcline;
    }

    header += "#define " + shaderStage + '\n';

    std::ostringstream out;
    if (spos == std::string::npos)
        out << source << "\n";
    else if (spos != 0)
        out << source.substr(0, spos);
    out << header;
    if (spos != std::string::npos)
    {
        out << "#line " << srcline << "\n";
        if (spos == 0)
            out << source;
        else
            out << source.substr(spos);
    }
    return out.str();
}

std::string GLSLShader::GetShaderStageName(GLint target)
{
    std::string shaderStage;
    switch(target)
    {
    case GL_VERTEX_SHADER_ARB  :    shaderStage = "Vertex"; break;
    case GL_FRAGMENT_SHADER_ARB:    shaderStage = "Fragment"; break;
#ifdef GL_GEOMETRY_SHADER_EXT
    case GL_GEOMETRY_SHADER_EXT:    shaderStage = "Geometry"; break;
#endif
#ifdef GL_TESS_CONTROL_SHADER
    case GL_TESS_CONTROL_SHADER:    shaderStage = "TessellationControl"; break;
#endif
#ifdef GL_TESS_EVALUATION_SHADER
    case GL_TESS_EVALUATION_SHADER: shaderStage = "TessellationEvaluation"; break;
#endif
    default:                        shaderStage = "Unknown"; break;
    }
    return shaderStage;
}

///	This function compiles a shader and check the log
bool GLSLShader::CompileShader(GLint target, const ShaderContents& shaderContent, const std::string& header)
{
    if (!GLSLIsSupported) return false;

    std::string shaderStage = GetShaderStageName(target);

    std::string source;
    //to get sure that the file has been loaded
    if(shaderContent.text.empty())
    {
        if(!shaderContent.filename.empty())
            source = LoadTextFile(shaderContent.filename);
        else
        {
            msg_error() << "No content has been given.";
            return false;
        }
    }
    else
        source = shaderContent.text;

    source = CombineHeaders(header, shaderStage + std::string("Shader"), source);

    GLhandleARB shader = glCreateShaderObjectARB(target);

    const char* src = source.c_str();

    glShaderSourceARB(shader, 1, &src, NULL);

    glCompileShaderARB(shader);

    GLint compiled = 0, length = 0, laux = 0;
    glGetObjectParameterivARB(shader, GL_OBJECT_COMPILE_STATUS_ARB, &compiled);
    if (!compiled)
        msg_error() << "ERROR: Compilation of " << shaderStage << " shader failed:";

    glGetObjectParameterivARB(shader, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
    if (length > 1)
    {
        if(!shaderContent.filename.empty())
            msg_error() << "From File: " << shaderContent.filename;
        else
            msg_error() << "From Source code (no filename given).";

        if (!header.empty())
            msg_error() << "Header:\n" << header;

        GLcharARB *logString = (GLcharARB *)malloc((length + 1) * sizeof(GLcharARB));
        glGetInfoLogARB(shader, length, &laux, logString);
        msg_error() << logString;
        free(logString);
    }
    if (compiled)
        m_hShaders[target] = shader;
    else
        glDeleteObjectARB(shader);
    return (compiled != 0);
}

///	This function loads a vertex and fragment shader file
void GLSLShader::InitShaders()
{
    if (!GLSLIsSupported) return;
    // Make sure the user passed in at least a vertex and fragment shader file
    if( !GetVertexShaderFileName().length() || !GetFragmentShaderFileName().length() )
    {
        if(m_hShaderContents.find(GL_VERTEX_SHADER_ARB) == m_hShaderContents.end()
            || m_hShaderContents.find(GL_FRAGMENT_SHADER_ARB) == m_hShaderContents.end())
        {
            msg_error() << "GLSLShader requires setting a VertexShader and a FragmentShader";
            return;
        }
    }

    // If any of our shader pointers are set, let's free them first.
    if(!m_hShaders.empty() || m_hProgramObject)
        Release();

    bool ready = true;

    for (std::map<GLint, ShaderContents>::const_iterator it = m_hShaderContents.begin(), itend = m_hShaderContents.end(); it != itend; ++it)
    {
        ready &= CompileShader(it->first, it->second, header);
    }

    if (!ready)
    {
        msg_error() << "SHADER compilation failed.";
        return;
    }

    // Next we create a program object to represent our shaders
    m_hProgramObject = glCreateProgramObjectARB();

    // We attach each shader we just loaded to our program object
    for (std::map<GLint,GLhandleARB>::const_iterator it = m_hShaders.begin(), itend = m_hShaders.end(); it != itend; ++it)
    {
        glAttachObjectARB(m_hProgramObject, it->second);
    }

#if defined(GL_EXT_geometry_shader4) && defined(SOFA_HAVE_GLEW)
    if (GetGeometryShaderID())
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
    if (!linked)
        msg_error() << "ERROR: Link of program shader failed:";

    glGetObjectParameterivARB(m_hProgramObject, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
    if (length > 1)
    {
        GLcharARB *logString = (GLcharARB *)malloc((length+1) * sizeof(GLcharARB));
        glGetInfoLogARB(m_hProgramObject, length, &laux, logString);
        msg_error() << logString;
        free(logString);
        for (std::map<GLint,ShaderContents>::const_iterator it = m_hShaderContents.begin(), itend = m_hShaderContents.end(); it != itend; ++it)
            if(!it->second.filename.empty())
                msg_error() << GetShaderStageName(it->first) << " shader file: " << it->second.filename ;
            else
                msg_error() << GetShaderStageName(it->first) << " , check your source file (not an external shader file)" ;

    }

    // Now, let's turn off the shader initially.
    glUseProgramObjectARB(0);
}

std::string GLSLShader::GetShaderFileName(GLint type) const
{
    std::map<GLint, ShaderContents>::const_iterator it = m_hShaderContents.find(type);
    return ((it != m_hShaderContents.end()) ? it->second.filename : std::string());
}

bool GLSLShader::IsSet(GLint type) const
{
    std::map<GLint, ShaderContents>::const_iterator it = m_hShaderContents.find(type);
    return it != m_hShaderContents.end();
}

std::string GLSLShader::GetShaderString(GLint type) const
{
    std::map<GLint, ShaderContents>::const_iterator it = m_hShaderContents.find(type);
    return ((it != m_hShaderContents.end()) ? it->second.text : std::string());
}

std::string GLSLShader::GetHeader() const
{
    return this->header;
}

GLhandleARB GLSLShader::GetShaderID(GLint type) const
{
    std::map<GLint,GLhandleARB>::const_iterator it = m_hShaders.find(type);
    return ((it != m_hShaders.end()) ? it->second : 0);
}

void GLSLShader::SetInt(GLint variable, int newValue)                              { if (variable!=-1) glUniform1iARB(variable, newValue);       }
void GLSLShader::SetInt2(GLint variable, int i1, int i2)                           { if (variable!=-1) glUniform2iARB(variable, i1, i2);         }
void GLSLShader::SetInt3(GLint variable, int i1, int i2, int i3)                   { if (variable!=-1) glUniform3iARB(variable, i1, i2, i3);     }
void GLSLShader::SetInt4(GLint variable, int i1, int i2, int i3, int i4)           { if (variable!=-1) glUniform4iARB(variable, i1, i2, i3, i4); }
void GLSLShader::SetFloat(GLint variable, float newValue)                          { if (variable!=-1) glUniform1fARB(variable, newValue);       }
void GLSLShader::SetFloat2(GLint variable, float v0, float v1)                     { if (variable!=-1) glUniform2fARB(variable, v0, v1);         }
void GLSLShader::SetFloat3(GLint variable, float v0, float v1, float v2)
{
    if (variable!=-1)
        glUniform3fARB(variable, v0, v1, v2);
}
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
    for (std::map<GLint,GLhandleARB>::const_iterator it = m_hShaders.begin(), itend = m_hShaders.end(); it != itend; ++it)
    {
        GLhandleARB shader = it->second;
        if (shader && m_hProgramObject)
            glDetachObjectARB(m_hProgramObject, shader);
        if (shader)
            glDeleteObjectARB(shader);
    }
    m_hShaders.clear();

    // If our program object pointer is valid, free it
    if(m_hProgramObject)
    {
        glDeleteObjectARB(m_hProgramObject);
        m_hProgramObject = 0;
    }
}

} // namespace gl

} // namespace helper

} // namespace sofa
