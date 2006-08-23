//*******************************************************************//
//	OpenGL Shader class 											 //
//                                                                   //
//	Based on code from Ben Humphrey	/ digiben@gametutorilas.com		 //
//																	 //
//*******************************************************************//

#include "GLshader.h"


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

bool InitGLSL()
{
    // This grabs a list of all the video card's extensions it supports
    char *szGLExtensions = (char*)glGetString(GL_EXTENSIONS);

    // Make sure find the GL_ARB_shader_objects extension so we can use shaders.
    if(!strstr(szGLExtensions, "GL_ARB_shader_objects"))
    {
        printf("GL_ARB_shader_objects extension not supported!\n");
        return false;
    }

    // Make sure we support the GLSL shading language 1.0
    if(!strstr(szGLExtensions, "GL_ARB_shading_language_100"))
    {
        printf("GL_ARB_shading_language_100 extension not supported!\n");
        return false;
    }

    // Now let's set all of our function pointers for our extension functions
    glCreateShaderObjectARB = (PFNGLCREATESHADEROBJECTARBPROC)wglGetProcAddress("glCreateShaderObjectARB");
    glShaderSourceARB = (PFNGLSHADERSOURCEARBPROC)wglGetProcAddress("glShaderSourceARB");
    glCompileShaderARB = (PFNGLCOMPILESHADERARBPROC)wglGetProcAddress("glCompileShaderARB");
    glCreateProgramObjectARB = (PFNGLCREATEPROGRAMOBJECTARBPROC)wglGetProcAddress("glCreateProgramObjectARB");
    glAttachObjectARB = (PFNGLATTACHOBJECTARBPROC)wglGetProcAddress("glAttachObjectARB");
    glLinkProgramARB = (PFNGLLINKPROGRAMARBPROC)wglGetProcAddress("glLinkProgramARB");
    glUseProgramObjectARB = (PFNGLUSEPROGRAMOBJECTARBPROC)wglGetProcAddress("glUseProgramObjectARB");
    glUniform1iARB = (PFNGLUNIFORM1IARBPROC)wglGetProcAddress("glUniform1iARB");
    glUniform1fARB = (PFNGLUNIFORM1FARBPROC)wglGetProcAddress("glUniform1fARB");
    glUniform2fARB = (PFNGLUNIFORM2FARBPROC)wglGetProcAddress("glUniform2fARB");
    glUniform3fARB = (PFNGLUNIFORM3FARBPROC)wglGetProcAddress("glUniform3fARB");
    glUniform4fARB = (PFNGLUNIFORM4FARBPROC)wglGetProcAddress("glUniform4fARB");
    glGetUniformLocationARB = (PFNGLGETUNIFORMLOCATIONARBPROC)wglGetProcAddress("glGetUniformLocationARB");
    glDetachObjectARB = (PFNGLDETACHOBJECTARBPROC)wglGetProcAddress("glDetachObjectARB");
    glDeleteObjectARB  = (PFNGLDELETEOBJECTARBPROC)wglGetProcAddress("glDeleteObjectARB");
    glActiveTextureARB = (PFNGLACTIVETEXTUREARBPROC)wglGetProcAddress("glActiveTextureARB");
    glMultiTexCoord2fARB = (PFNGLMULTITEXCOORD2FARBPROC)wglGetProcAddress("glMultiTexCoord2fARB");
    glGetObjectParameterivARB = (PFNGLGETOBJECTPARAMETERIV)wglGetProcAddress("glGetObjectParameterivARB");
    glGetInfoLogARB = (PFNGLGETINFOLOGARBPROC)wglGetProcAddress("glGetInfoLogARB");

    // Return a success!
    return true;
}


///////////////////////////////// LOAD TEXT FILE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////	This function loads and returns a text file for our shaders
/////
///////////////////////////////// LOAD TEXT FILE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

string CShader::LoadTextFile(string strFile)
{
    // Open the file passed in
    ifstream fin(strFile.c_str());

    // Make sure we opened the file correctly
    if(!fin)
        return "";

    string strLine = "";
    string strText = "";

    // Go through and store each line in the text file within a "string" object
    while(getline(fin, strLine))
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
/////	This function loads a vertex and fragment shader file
/////
///////////////////////////////// INIT SHADERS \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

void CShader::InitShaders(string strVertex, string strFragment)
{
    // These will hold the shader's text file data
    string strVShader, strFShader;

    // Make sure the user passed in a vertex and fragment shader file
    if(!strVertex.length() || !strFragment.length())
        return;

    // If any of our shader pointers are set, let's free them first.
    if(m_hVertexShader || m_hFragmentShader || m_hProgramObject)
        Release();

    // Here we get a pointer to our vertex and fragment shaders
    m_hVertexShader = glCreateShaderObjectARB(GL_VERTEX_SHADER_ARB);
    m_hFragmentShader = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);

    // Now we load the shaders from the respective files and store it in a string.
    strVShader = LoadTextFile(strVertex.c_str());
    strFShader = LoadTextFile(strFragment.c_str());

    // Do a quick switch so we can do a double pointer below
    const char *szVShader = strVShader.c_str();
    const char *szFShader = strFShader.c_str();

    // Now this assigns the shader text file to each shader pointer
    glShaderSourceARB(m_hVertexShader, 1, &szVShader, NULL);
    glShaderSourceARB(m_hFragmentShader, 1, &szFShader, NULL);

    // Now we actually compile the shader's code
    glCompileShaderARB(m_hVertexShader);
    glCompileShaderARB(m_hFragmentShader);

    // Next we create a program object to represent our shaders
    m_hProgramObject = glCreateProgramObjectARB();

    // We attach each shader we just loaded to our program object
    glAttachObjectARB(m_hProgramObject, m_hVertexShader);
    glAttachObjectARB(m_hProgramObject, m_hFragmentShader);

    // Our last init function is to link our program object with OpenGL
    glLinkProgramARB(m_hProgramObject);

    int logLength = 0;
    glGetObjectParameterivARB(m_hProgramObject, GL_OBJECT_INFO_LOG_LENGTH_ARB, &logLength);

    if (logLength > 1)
    {
        char *szLog = (char*)malloc(logLength);
        int writtenLength = 0;

        glGetInfoLogARB(m_hProgramObject, logLength, &writtenLength, szLog);
        printf("ERROR during shader initialization...\n");

        free(szLog);
    }

    // Now, let's turn off the shader initially.
    glUseProgramObjectARB(0);
}


///////////////////////////////// GET VARIABLE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////	This function returns a variable ID for a shader variable
/////
///////////////////////////////// GET VARIABLE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

GLint CShader::GetVariable(string strVariable)
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
        m_hVertexShader = NULL;
    }

    // If our fragment shader pointer is valid, free it
    if(m_hFragmentShader)
    {
        glDetachObjectARB(m_hProgramObject, m_hFragmentShader);
        glDeleteObjectARB(m_hFragmentShader);
        m_hFragmentShader = NULL;
    }

    // If our program object pointer is valid, free it
    if(m_hProgramObject)
    {
        glDeleteObjectARB(m_hProgramObject);
        m_hProgramObject = NULL;
    }
}


/////////////////////////////////////////////////////////////////////////////////
//
// * QUICK NOTES *
//
// Nothing new was added to this file for this tutorial.
//
//
// © 2000-2005 GameTutorials
