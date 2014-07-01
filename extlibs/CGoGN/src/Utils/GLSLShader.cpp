/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
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
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#define EXPORTING 1

#include "Utils/GLSLShader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "Utils/cgognStream.h"

#include "glm/gtx/inverse_transpose.hpp"


namespace CGoGN
{

namespace Utils
{

unsigned int GLSLShader::CURRENT_OGL_VERSION = 2;

std::string GLSLShader::DEFINES_GL2=\
"#version 110\n"
"#define PRECISON float pipo_PRECISION\n"
"#define ATTRIBUTE attribute\n"
"#define VARYING_VERT varying\n"
"#define VARYING_FRAG varying\n"
"#define FRAG_OUT_DEF float pipo_FRAGDEF\n"
"#define INVARIANT_POS float pipo_INVARIANT\n";


std::string GLSLShader::DEFINES_GL3=\
"#version 150\n"
"#define PRECISON precision highp float\n"
"#define ATTRIBUTE in\n"
"#define VARYING_VERT smooth out\n"
"#define VARYING_FRAG smooth in\n"
"#define FRAG_OUT_DEF out vec4 gl_FragColor\n"
"#define INVARIANT_POS invariant gl_Position\n";


std::string* GLSLShader::DEFINES_GL = NULL;

std::vector<std::string> GLSLShader::m_pathes;

std::set< std::pair<void*, GLSLShader*> > GLSLShader::m_registeredShaders;


//glm::mat4* GLSLShader::s_current_matrices=NULL;
Utils::GL_Matrices* GLSLShader::s_current_matrices=NULL;

GLSLShader::GLSLShader() :
	m_vertex_shader_source(NULL),
	m_fragment_shader_source(NULL),
	m_geom_shader_source(NULL)
{
	*m_vertex_shader_object = 0;
	*m_fragment_shader_object = 0;
	*m_geom_shader_object = 0;
	*m_program_object = 0;
	*m_uniMat_Proj = -1;
	*m_uniMat_Model = -1;
	*m_uniMat_ModelProj = -1;
	*m_uniMat_Normal = -1;

	if (DEFINES_GL == NULL)
		DEFINES_GL = &DEFINES_GL2;

	m_nbMaxVertices = 16;
}

void GLSLShader::registerShader(void* ptr, GLSLShader* shader)
{
	m_registeredShaders.insert(std::pair<void*,GLSLShader*>(ptr, shader));
}

void GLSLShader::unregisterShader(void* ptr, GLSLShader* shader)
{
	m_registeredShaders.erase(std::pair<void*,GLSLShader*>(ptr, shader));
}

std::string GLSLShader::defines_Geom(const std::string& primitivesIn, const std::string& primitivesOut, int maxVert)
{
	if (CURRENT_OGL_VERSION == 3)
	{
		std::string str("#version 150\n");
		str.append("precision highp float;\n");
		str.append("layout (");
		str.append(primitivesIn);
		str.append(") in;\n");

		str.append("layout (");
		str.append(primitivesOut);
		str.append(", max_vertices = ");
		std::stringstream ss;
		ss << maxVert;
		str.append(ss.str());
		str.append(") out;\n");
		str.append("#define VARYING_IN in\n");
		str.append("#define VARYING_OUT smooth out\n");
		str.append("#define POSITION_IN(X) gl_in[X].gl_Position\n");
		str.append("#define NBVERTS_IN gl_in.length()\n");
		return str;
	}
	else
	{
		std::string str("#version 110\n");
		str.append("#extension GL_EXT_geometry_shader4 : enable\n");
		str.append("#define PRECISON float pipo_PRECISION\n");
		str.append("#define ATTRIBUTE attribute\n");
		str.append("#define VARYING_IN varying in\n");
		str.append("#define VARYING_OUT varying out\n");
		str.append("#define POSITION_IN(X) gl_PositionIn[X]\n");
		str.append("#define NBVERTS_IN gl_VerticesIn\n");
		return str;
	}
}

bool GLSLShader::areGeometryShadersSupported()
{
	if (!glewGetExtension("GL_EXT_geometry_shader4"))
		return false;
	return true;
}

bool GLSLShader::areShadersSupported()
{
	if ( ! glewGetExtension("GL_ARB_vertex_shader")) return false;
	if ( ! glewGetExtension("GL_ARB_fragment_shader")) return false;
	if ( ! glewGetExtension("GL_ARB_shader_objects")) return false;
	if ( ! glewGetExtension("GL_ARB_shading_language_100")) return false;

	return true;
}

bool GLSLShader::areVBOSupported()
{
	if (!glewGetExtension("GL_ARB_vertex_buffer_object"))
		return false;
	return true;
}

char* GLSLShader::loadSourceFile(const std::string& filename)
{
	std::ifstream	file;
	int				file_size;
	char*			shader_source;

	/*** opening file ***/
	file.open( filename.c_str() , std::ios::in | std::ios::binary );

	if( !file.good() )
	{
		CGoGNerr << "ERROR - GLSLShader::loadSourceFile() - unable to open the file " << filename << "." << CGoGNendl;
		return NULL;
	}

	/*** reading file ***/
	try
	{
		/* get file size */
		file.seekg( 0, std::ios::end );
		file_size = file.tellg();
		file.seekg( 0, std::ios::beg );

		/* allocate shader source table */
		shader_source = new char [ file_size+1 ];

		/* read source file */
		file.read( shader_source, file_size );
		shader_source[ file_size ] = '\0';
	}
	catch( std::exception& io_exception )
	{
		CGoGNerr << "ERROR - GLSLShader::loadSourceFile() - " << io_exception.what() << CGoGNendl;
		file.close();
		return NULL;
	}

	/*** termination ***/
	file.close();
	return shader_source;
}

bool GLSLShader::loadVertexShader(  const std::string& filename )
{
	bool	flag;
//	char	*vertex_shader_source;

	if (m_vertex_shader_source)
		delete [] m_vertex_shader_source;
	m_vertex_shader_source = NULL;

	m_vertex_shader_source = loadSourceFile( filename );

	if( !m_vertex_shader_source )
	{
		CGoGNerr << "ERROR - GLSLShader::loadVertexShader() - error occured while loading source file." << CGoGNendl;
		return false;
	}


	flag = loadVertexShaderSourceString( m_vertex_shader_source );
//	delete [] vertex_shader_source;

	return flag;
}

bool GLSLShader::loadFragmentShader(const std::string& filename )
{
	bool	flag;
//	char	*fragment_shader_source;

	if (m_fragment_shader_source)
		delete [] m_fragment_shader_source;
	m_fragment_shader_source = NULL;

	m_fragment_shader_source = loadSourceFile( filename );

	if( !m_fragment_shader_source )
	{
		CGoGNerr << "ERROR - GLSLShader::loadFragmentShader() - error occured while loading source file." << CGoGNendl;
		return false;
	}

	flag = loadFragmentShaderSourceString( m_fragment_shader_source );
//	delete [] fragment_shader_source;


	return flag;
}

bool GLSLShader::loadGeometryShader(const std::string& filename )
{
	bool	flag;
//	char	*geom_shader_source;

	if (m_geom_shader_source)
		delete [] m_geom_shader_source;

	m_geom_shader_source = loadSourceFile( filename );

	if( !m_geom_shader_source )
	{
		CGoGNerr << "ERROR - GLSLShader::loadGeometryShader() - error occured while loading source file." << CGoGNendl;
		return false;
	}

	flag = loadGeometryShaderSourceString( m_geom_shader_source );
//	delete [] geom_shader_source;

	return flag;
}

bool GLSLShader::logError(GLuint handle, const std::string& nameSrc, const char *src)
{
	char *info_log;
	info_log = getInfoLog( handle );
	if (info_log!=NULL)
	{
		CGoGNerr << "============================================================================" << CGoGNendl;
		CGoGNerr << "Error in " << nameSrc << CGoGNendl;
		CGoGNerr << "----------------------------------------------------------------------------" << CGoGNendl;
		char line[256];
		int ln=1;
		std::stringstream ss(src);
		do
		{
			ss.getline(line,256);
			std::cout << ln++ << ": "<< line<< std::endl;
		}while (!ss.eof());
		CGoGNerr << "----------------------------------------------------------------------------" << CGoGNendl;
		CGoGNerr << info_log;
		CGoGNerr << "============================================================================" << CGoGNendl;
		delete [] info_log;
		return false;
	}
	return true;
}


bool GLSLShader::loadVertexShaderSourceString( const char *vertex_shader_source )
{
	if (*m_vertex_shader_object==0)
	{
		glDeleteShader(*m_vertex_shader_object);
		*m_vertex_shader_object=0;
	}

	/*** create shader object ***/
	*m_vertex_shader_object = glCreateShader( GL_VERTEX_SHADER );

	if( !*m_vertex_shader_object )
	{
		CGoGNerr << "ERROR - GLSLShader::loadVertexShader() - unable to create shader object." << CGoGNendl;
		return false;
	}

	/*** load source file ***/
	if( !vertex_shader_source )
	{
		CGoGNerr << "ERROR - GLSLShader::loadVertexShader() - source string is empty." << CGoGNendl;

		glDeleteShader(*m_vertex_shader_object );
		*m_vertex_shader_object = 0;

		return false;
	}

	glShaderSource( *m_vertex_shader_object, 1, (const char**)&vertex_shader_source, NULL );

	/*** compile shader object ***/
	glCompileShader( *m_vertex_shader_object );

	if (!logError(*m_vertex_shader_object, m_nameVS, vertex_shader_source))
	{
		glDeleteShader( *m_vertex_shader_object );
		*m_vertex_shader_object = 0;
		return false;
	}

	/*** termination ***/
	return true;
}

bool GLSLShader::loadFragmentShaderSourceString( const char *fragment_shader_source )
{
	if (*m_fragment_shader_object==0)
	{
		glDeleteShader(*m_fragment_shader_object);
		*m_fragment_shader_object=0;
	}

	/*** create shader object ***/
	*m_fragment_shader_object = glCreateShader( GL_FRAGMENT_SHADER );

	if( !*m_fragment_shader_object )
	{
		CGoGNerr << "ERROR - GLSLShader::loadFragmentShader() - unable to create shader object." << CGoGNendl;
		return false;
	}

	/*** load source file ***/
	if( !fragment_shader_source )
	{
		CGoGNerr << "ERROR - GLSLShader::loadFragmentShader() - source string is empty." << CGoGNendl;

		glDeleteShader( *m_fragment_shader_object );
		*m_fragment_shader_object = 0;

		return false;
	}

	glShaderSource( *m_fragment_shader_object, 1, (const char**)&fragment_shader_source, NULL );

	/*** compile shader object ***/
	glCompileShader( *m_fragment_shader_object );

	if (!logError(*m_fragment_shader_object, m_nameFS, fragment_shader_source))
	{
		glDeleteShader( *m_fragment_shader_object );
		*m_fragment_shader_object = 0;
		return false;
	}


	/*** termination ***/
	return true;
}

bool GLSLShader::loadGeometryShaderSourceString( const char *geom_shader_source )
{
	if (*m_geom_shader_object==0)
	{
		glDeleteShader(*m_geom_shader_object);
		*m_geom_shader_object=0;
	}
	/*** create shader object ***/
	*m_geom_shader_object = glCreateShader(GL_GEOMETRY_SHADER_EXT);

	if( !*m_geom_shader_object )
	{
		CGoGNerr << "ERROR - GLSLShader::loadGeometryShader() - unable to create shader object." << CGoGNendl;
		return false;
	}

	/*** load source file ***/
	if( !geom_shader_source )
	{
		CGoGNerr << "ERROR - GLSLShader::loadGeometryShader() - source string is empty." << CGoGNendl;

		glDeleteShader( *m_geom_shader_object );
		*m_geom_shader_object = 0;

		return false;
	}

	glShaderSource( *m_geom_shader_object, 1, (const char**)&geom_shader_source, NULL );

	/*** compile shader object ***/
	glCompileShader( *m_geom_shader_object );

	if (!logError(*m_geom_shader_object, m_nameGS, geom_shader_source))
	{
		glDeleteShader( *m_geom_shader_object );
		*m_geom_shader_object = 0;
		return false;
	}


	/*** termination ***/
	return true;
}

char* GLSLShader::getInfoLog( GLuint obj )
{
	char	*info_log;
	int		info_log_length;
	int		length;

	glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &info_log_length);

	if (info_log_length <= 1)
		return NULL;

	info_log = new char [info_log_length];
	glGetShaderInfoLog( obj, info_log_length, &length, info_log );

	return info_log;
}



bool GLSLShader::create(GLint inputGeometryPrimitive,GLint outputGeometryPrimitive, int nb_max_vertices)
{
	int		status;
	char	*info_log;

	if (nb_max_vertices != -1)
		m_nbMaxVertices = nb_max_vertices;

	m_geom_inputPrimitives = inputGeometryPrimitive;
	m_geom_outputPrimitives = outputGeometryPrimitive;

	/*** check if shaders are loaded ***/
	if( !*m_vertex_shader_object || !*m_fragment_shader_object )
	{
		CGoGNerr << "ERROR - GLSLShader::create() - shaders are not defined." << CGoGNendl;
		return false;
	}

	/*** create program object ***/
	m_program_object = glCreateProgram();

	if( !*m_program_object )
	{
		CGoGNerr << "ERROR - GLSLShader::create() - unable to create program object." << CGoGNendl;
		return false;
	}

	/*** attach shaders to program object ***/
	glAttachShader( *m_program_object, *m_vertex_shader_object );
	glAttachShader( *m_program_object, *m_fragment_shader_object );
	if (*m_geom_shader_object)
	{
		glAttachShader( *m_program_object, *m_geom_shader_object );

		glProgramParameteriEXT(*m_program_object, GL_GEOMETRY_INPUT_TYPE_EXT, inputGeometryPrimitive);
		glProgramParameteriEXT(*m_program_object, GL_GEOMETRY_OUTPUT_TYPE_EXT, outputGeometryPrimitive);
		glProgramParameteriEXT(*m_program_object, GL_GEOMETRY_VERTICES_OUT_EXT, m_nbMaxVertices);
	}

	/*** link program object ***/
	glLinkProgram( *m_program_object );

	glGetProgramiv( *m_program_object, GL_OBJECT_LINK_STATUS_ARB, &status );
	if( !status )
	{
		CGoGNerr << "ERROR - GLSLShader::create() - error occured while linking shader program." << CGoGNendl;
		info_log = getInfoLog( *m_program_object );
		CGoGNerr << "  LINK " << info_log << CGoGNendl;
		delete [] info_log;

		glDetachShader( *m_program_object, *m_vertex_shader_object );
		glDetachShader( *m_program_object, *m_fragment_shader_object );
		if (*m_geom_shader_object)
			glDetachShader( *m_program_object, *m_geom_shader_object );
		glDeleteShader( *m_program_object );
		*m_program_object = 0;

		return false;
	}

	*m_uniMat_Proj		= glGetUniformLocation(*m_program_object, "ProjectionMatrix");
	*m_uniMat_Model		= glGetUniformLocation(*m_program_object, "ModelViewMatrix");
	*m_uniMat_ModelProj	= glGetUniformLocation(*m_program_object, "ModelViewProjectionMatrix");
	*m_uniMat_Normal	= glGetUniformLocation(*m_program_object, "NormalMatrix");

	return true;
}



bool GLSLShader::changeNbMaxVertices(int nb_max_vertices)
{
	m_nbMaxVertices = nb_max_vertices;
	if (*m_geom_shader_object)
	{
		glProgramParameteriEXT(*m_program_object,GL_GEOMETRY_VERTICES_OUT_EXT,m_nbMaxVertices);
		// need to relink
		return true;
	}
	return false;
}







bool GLSLShader::link()
{
	int		status;
	char	*info_log;

	/*** link program object ***/
	glLinkProgram( *m_program_object );

	glGetProgramiv( *m_program_object, GL_OBJECT_LINK_STATUS_ARB, &status );
	if( !status )
	{
		CGoGNerr << "ERROR - GLSLShader::create() - error occured while linking shader program." << CGoGNendl;
		info_log = getInfoLog( *m_program_object );
		CGoGNerr << "  LINK " << info_log << CGoGNendl;
		delete [] info_log;

		glDetachShader( *m_program_object, *m_vertex_shader_object );
		glDetachShader( *m_program_object, *m_fragment_shader_object );
		if (*m_geom_shader_object)
			glDetachShader( *m_program_object, *m_geom_shader_object );
		glDeleteShader( *m_program_object );
		*m_program_object = 0;

		return false;
	}

	return true;
}

bool GLSLShader::bind() const
{
	if( *m_program_object )
	{
		glUseProgram( *m_program_object );
		return true;
	}
	else
		return false;
}

void GLSLShader::unbind() const
{
	if( *m_program_object )
	{
		glUseProgram( 0 );
	}
}

bool GLSLShader::isBinded()
{
	if (*m_program_object == 0)
		return false;

	GLint po;
	glGetIntegerv(GL_CURRENT_PROGRAM,&po);
	return ( *m_program_object == po );
}

GLSLShader::~GLSLShader()
{
	if( *m_program_object )
	{
		unbind();

		if( *m_vertex_shader_object )
		{
			glDetachShader( *m_program_object, *m_vertex_shader_object );
			glDeleteShader( *m_vertex_shader_object );
		}
		if( *m_fragment_shader_object )
		{
			glDetachShader( *m_program_object, *m_fragment_shader_object );
			glDeleteShader( *m_fragment_shader_object );
		}
		if (*m_geom_shader_object)
		{
			glDetachShader( *m_program_object, *m_geom_shader_object );
			glDeleteShader( *m_geom_shader_object );
		}

		glDeleteShader( *m_program_object );
	}

	if (m_vertex_shader_source != NULL)
		delete[] m_vertex_shader_source;
	if (m_fragment_shader_source != NULL)
		delete[] m_fragment_shader_source;
	if (m_geom_shader_source != NULL)
		delete[] m_geom_shader_source;

//	m_registeredShaders.erase(this);
}

std::string GLSLShader::findFile(const std::string filename)
{
	// cherche d'abord dans le repertoire courant
	std::ifstream file;
	file.open(filename.c_str(),std::ios::in );
	if (!file.fail())
	{
		file.close();
		return filename;
	}
	file.close();

	for (std::vector<std::string>::const_iterator ipath = m_pathes.begin(); ipath != m_pathes.end(); ++ipath)
	{
		std::string st(*ipath);
		st.append(filename);

		std::ifstream file2;
		file2.open(st.c_str(),std::ios::in);
		if (!file2.fail())
		{
			file2.close();
			return st;
		}
	}

	// LA MACRO SHADERPATH contient le chemin du repertoire qui contient les fichiers textes
	std::string st(SHADERPATH);
	st.append(filename);

	std::ifstream file2; // on ne peut pas réutiliser file ????
	file2.open(st.c_str(),std::ios::in);
	if (!file2.fail())
	{
		file2.close();
		return st;
	}

	return filename;
}

bool GLSLShader::init()
{
#ifndef GLEW_MX
	GLenum error = glewInit();

	if (error != GLEW_OK)
		CGoGNerr << "Error: " << glewGetErrorString(error) << CGoGNendl;
	else
		CGoGNout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << CGoGNendl;

	if (!areVBOSupported())
		CGoGNout << "VBO not supported !" << CGoGNendl;

	if(!areShadersSupported()) {
		CGoGNout << "Shaders not supported !" << CGoGNendl;
		return false;
	}
#endif
	return true;

}

bool GLSLShader::loadShaders(const std::string& vs, const std::string& ps)
{
	m_nameVS = vs;
	m_nameFS = ps;

	std::string vss = findFile(vs);
	if(!loadVertexShader(vss)) return false;
	
	std::string pss = findFile(ps);
	if(!loadFragmentShader(pss)) return false;

	if(!create()) {
		CGoGNout << "Unable to create the shaders !" << CGoGNendl;
		return false;
	}
	CGoGNout << "Shaders loaded (" << vs << "," << ps << ")" << CGoGNendl;
	return true; 
}

bool GLSLShader::loadShaders(const std::string& vs, const std::string& ps, const std::string& gs, GLint inputGeometryPrimitive,GLint outputGeometryPrimitive, int nb_max_vertices)
{
	m_nameVS = vs;
	m_nameFS = ps;
	m_nameGS = gs;

	std::string vss = findFile(vs);
	if(!loadVertexShader(vss)) return false;

	std::string pss = findFile(ps);
	if(!loadFragmentShader(pss)) return false;

	std::string gss = findFile(gs);
	bool geomShaderLoaded = loadGeometryShader(gss);

	if (!geomShaderLoaded)
	{
		CGoGNerr << "Error while loading geometry shader" << CGoGNendl;
	}

	if(!create(inputGeometryPrimitive,outputGeometryPrimitive,nb_max_vertices))
	{
		CGoGNout << "Unable to create the shaders !" << CGoGNendl;
		return false;
	}

	CGoGNout << "Shaders loaded (" << vs << "," << ps << "," << gs <<")" << CGoGNendl;
	return true;
}

bool GLSLShader::loadShadersFromMemory(const char* vs, const char* fs)
{
	if (m_vertex_shader_source)
		delete [] m_vertex_shader_source;
	m_vertex_shader_source = NULL;

	unsigned int sz = strlen(vs);
	m_vertex_shader_source = new char[sz+1];
	strcpy(m_vertex_shader_source, vs);

	if (m_fragment_shader_source)
		delete [] m_fragment_shader_source;

	sz = strlen(fs);
	m_fragment_shader_source = new char[sz+1];
	strcpy(m_fragment_shader_source, fs);

	if(!loadVertexShaderSourceString(vs))
		return false;

	if(!loadFragmentShaderSourceString(fs))
		return false;

	if(!create())
	{
		CGoGNout << "Unable to create the shaders !" << CGoGNendl;
		return false;
	}
	return true;
}

bool GLSLShader::loadShadersFromMemory(const char* vs, const char* fs, const char* gs, GLint inputGeometryPrimitive,GLint outputGeometryPrimitive, int nb_max_vertices)
{
	if (m_vertex_shader_source)
		delete [] m_vertex_shader_source;
	m_vertex_shader_source = NULL;

	unsigned int sz = strlen(vs);
	m_vertex_shader_source = new char[sz+1];
	strcpy(m_vertex_shader_source,vs);

	if (m_fragment_shader_source)
		delete [] m_fragment_shader_source;

	sz = strlen(fs);
	m_fragment_shader_source = new char[sz+1];
	strcpy(m_fragment_shader_source,fs);

	if (m_geom_shader_source)
		delete [] m_geom_shader_source;

	sz = strlen(gs);
	m_geom_shader_source = new char[sz+1];
	strcpy(m_geom_shader_source,gs);

	if(!loadVertexShaderSourceString(vs))
		return false;

	if(!loadFragmentShaderSourceString(fs))
		return false;

	if(!loadGeometryShaderSourceString(gs))
		return false;

	if(!create(inputGeometryPrimitive, outputGeometryPrimitive, nb_max_vertices))
	{
		CGoGNout << "Unable to create the shaders !" << CGoGNendl;
		return false;
	}

	return true;
}

bool GLSLShader::reloadVertexShaderFromMemory(const char* vs)
{
	if (m_vertex_shader_source)
		delete [] m_vertex_shader_source;
	m_vertex_shader_source = NULL;

	unsigned int sz = strlen(vs);
	m_vertex_shader_source = new char[sz+1];

	strcpy(m_vertex_shader_source,vs);

	return true;
}

bool GLSLShader::reloadFragmentShaderFromMemory(const char* fs)
{
	if (m_fragment_shader_source)
		delete [] m_fragment_shader_source;

	unsigned int sz = strlen(fs);
	m_fragment_shader_source = new char[sz+1];
	strcpy(m_fragment_shader_source,fs);

	return true;
}

bool GLSLShader::reloadGeometryShaderFromMemory(const char* gs)
{
	if (m_geom_shader_source)
		delete [] m_geom_shader_source;

	unsigned int sz = strlen(gs);
	m_geom_shader_source = new char[sz+1];
	strcpy(m_geom_shader_source,gs);

	return true;
}

bool GLSLShader::recompile()
{
	if (m_vertex_shader_source)
		if(!loadVertexShaderSourceString(m_vertex_shader_source)) return false;
	if (m_fragment_shader_source)
		if(!loadFragmentShaderSourceString(m_fragment_shader_source)) return false;
	if (m_geom_shader_source)
		if(!loadGeometryShaderSourceString(m_geom_shader_source)) return false;

	if(!create(m_geom_inputPrimitives,m_geom_outputPrimitives))
	{
		CGoGNerr << "Unable to create the shaders !" << CGoGNendl;
		return false;
	}

	*m_uniMat_Proj		= glGetUniformLocation(*m_program_object,"ProjectionMatrix");
	*m_uniMat_Model		= glGetUniformLocation(*m_program_object,"ModelViewMatrix");
	*m_uniMat_ModelProj	= glGetUniformLocation(*m_program_object,"ModelViewProjectionMatrix");
	*m_uniMat_Normal	= glGetUniformLocation(*m_program_object,"NormalMatrix");

	restoreUniformsAttribs();

	updateClippingUniforms();

	return true;
}

bool GLSLShader::validateProgram()
{
	if(!*m_program_object)
		return false;

	glValidateProgram(*m_program_object);
	GLint Result = GL_FALSE;
	glGetProgramiv(*m_program_object, GL_VALIDATE_STATUS, &Result);

	if(Result == GL_FALSE)
	{
		CGoGNout << "Validate program:" << CGoGNendl;
		int InfoLogLength;
		glGetProgramiv(*m_program_object, GL_INFO_LOG_LENGTH, &InfoLogLength);
		std::vector<char> Buffer(InfoLogLength);
		glGetProgramInfoLog(*m_program_object, InfoLogLength, NULL, &Buffer[0]);
		CGoGNout <<  &(Buffer[0]) << CGoGNendl;
		return false;
	}

	return true;
}

bool GLSLShader::checkProgram()
{
	GLint Result = GL_FALSE;
	glGetProgramiv(*m_program_object, GL_LINK_STATUS, &Result);

	int InfoLogLength;
	glGetProgramiv(*m_program_object, GL_INFO_LOG_LENGTH, &InfoLogLength);
	std::vector<char> Buffer(std::max(InfoLogLength, int(1)));
	glGetProgramInfoLog(*m_program_object, InfoLogLength, NULL, &Buffer[0]);
	CGoGNout << &Buffer[0] << CGoGNendl;

	return Result == GL_TRUE;
}

bool GLSLShader::checkShader(int shaderType)
{
	GLint Result = GL_FALSE;
	int InfoLogLength;
	GLuint id;

	switch(shaderType)
	{
	case VERTEX_SHADER:
		id = *m_vertex_shader_object;
		break;
	case FRAGMENT_SHADER:
		id = *m_fragment_shader_object;
		break;
	case GEOMETRY_SHADER:
		id = *m_geom_shader_object;
		break;
	default:
		CGoGNerr << "Error unkown shader type" << CGoGNendl;
		return false;
		break;
	}

	glGetShaderiv(id, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(id, GL_INFO_LOG_LENGTH, &InfoLogLength);
	std::vector<char> Buffer(InfoLogLength);
	glGetShaderInfoLog(id, InfoLogLength, NULL, &Buffer[0]);
	CGoGNout << &Buffer[0] << CGoGNendl;

	return Result == GL_TRUE;
}

void GLSLShader::bindAttrib(unsigned int att, const char* name) const
{
	glBindAttribLocation(*m_program_object, att, name);
}

void GLSLShader::addPathFileSeach(const std::string& path)
{
	m_pathes.push_back(path);
}

unsigned int GLSLShader::bindVA_VBO(const std::string& name, VBO* vbo)
{
	GLint idVA = glGetAttribLocation(*(this->m_program_object), name.c_str());
	//valid ?
	if (idVA < 0)
	{
		CGoGNerr << "GLSLShader: Attribute " << name << " does not exist in shader" << CGoGNendl;
		return idVA;
	}
	// search if name already exist
	for (std::vector<VAStr>::iterator it = m_va_vbo_binding.begin(); it != m_va_vbo_binding.end(); ++it)
	{
		if (it->va_id == idVA)
		{
			it->vbo_ptr = vbo;
			return (it - m_va_vbo_binding.begin());
		}
	}
	// new one:
	VAStr temp;
	temp.va_id = idVA;
	temp.vbo_ptr = vbo;
	m_va_vbo_binding.push_back(temp);
	return (m_va_vbo_binding.size() -1);
}

void GLSLShader::changeVA_VBO(unsigned int id, VBO* vbo)
{
	m_va_vbo_binding[id].vbo_ptr = vbo;
}

void GLSLShader::unbindVA(const std::string& name)
{
	GLint idVA = glGetAttribLocation(*(this->m_program_object), name.c_str());
	//valid ?
	if (idVA < 0)
	{
		CGoGNerr << "GLSLShader: Attribute " << name << " does not exist in shader, not unbinded" << CGoGNendl;
		return;
	}
	// search if name already exist
	unsigned int nb = m_va_vbo_binding.size();
	for (unsigned int i = 0; i < nb; ++i)
	{
		if (m_va_vbo_binding[i].va_id == idVA)
		{
			if (i != (nb-1))
				m_va_vbo_binding[i] = m_va_vbo_binding[nb-1];
			m_va_vbo_binding.pop_back();
			return;
		}
	}
	CGoGNerr << "GLSLShader: Attribute "<<name<< " not binded"<< CGoGNendl;
}

void GLSLShader::setCurrentOGLVersion(unsigned int version)
{
	CURRENT_OGL_VERSION = version;
	switch(version)
	{
	case 2:
		DEFINES_GL = &DEFINES_GL2;
		break;
	case 3:
		DEFINES_GL = &DEFINES_GL3;
		break;
	}
}

/**
 * update projection, modelview, ... matrices
 */
void GLSLShader::updateMatrices(const glm::mat4& projection, const glm::mat4& modelview)
{
	this->bind();

	if (*m_uniMat_Proj >= 0)
		glUniformMatrix4fv(*m_uniMat_Proj, 1, false, &projection[0][0]);

	if (*m_uniMat_Model >= 0)
		glUniformMatrix4fv(*m_uniMat_Model,	1, false, &modelview[0][0]);

	if (*m_uniMat_ModelProj >= 0)
	{
		glm::mat4 PMV = projection * modelview;
		glUniformMatrix4fv(*m_uniMat_ModelProj,	1 , false, &PMV[0][0]);
	}

	if (*m_uniMat_Normal >= 0)
	{
		glm::mat4 normalMatrix = glm::gtx::inverse_transpose::inverseTranspose(modelview);
		glUniformMatrix4fv(*m_uniMat_Normal, 	1 , false, &normalMatrix[0][0]);
	}

	this->unbind();
}

void GLSLShader::updateMatrices(const glm::mat4& projection, const glm::mat4& modelview, const glm::mat4& PMV, const glm::mat4& normalMatrix)
{
	this->bind();

	if (*m_uniMat_Proj >= 0)
		glUniformMatrix4fv(*m_uniMat_Proj, 1, false, &projection[0][0]);

	if (*m_uniMat_Model >= 0)
		glUniformMatrix4fv(*m_uniMat_Model,	1, false, &modelview[0][0]);

	if (*m_uniMat_ModelProj >= 0)
		glUniformMatrix4fv(*m_uniMat_ModelProj,	1 , false, &PMV[0][0]);

	if (*m_uniMat_Normal >= 0)
		glUniformMatrix4fv(*m_uniMat_Normal, 	1 , false, &normalMatrix[0][0]);

	this->unbind();
}




void GLSLShader::enableVertexAttribs(unsigned int stride, unsigned int begin) const
{
	this->bind();
	for (std::vector<Utils::GLSLShader::VAStr>::const_iterator it = m_va_vbo_binding.begin(); it != m_va_vbo_binding.end(); ++it)
	{
		glBindBuffer(GL_ARRAY_BUFFER, it->vbo_ptr->id());
		glEnableVertexAttribArray(it->va_id);
		glVertexAttribPointer(it->va_id, it->vbo_ptr->dataSize(), GL_FLOAT, false, stride, (const GLvoid*)((unsigned long)(begin)));
	}
//	this->unbind();
}

void GLSLShader::disableVertexAttribs() const
{
//	this->bind();
	for (std::vector<Utils::GLSLShader::VAStr>::const_iterator it = m_va_vbo_binding.begin(); it != m_va_vbo_binding.end(); ++it)
		glDisableVertexAttribArray(it->va_id);
	this->unbind();
}


void GLSLShader::updateCurrentMatrices()
{
	glm::mat4 model(currentModelView());
	model *= currentTransfo();

	currentPMV() = currentProjection() * model;
	currentNormalMatrix() = glm::gtx::inverse_transpose::inverseTranspose(model);

	for(std::set< std::pair<void*, GLSLShader*> >::iterator it = m_registeredShaders.begin(); it != m_registeredShaders.end(); ++it)
		it->second->updateMatrices(currentProjection(), model, currentPMV(), currentNormalMatrix());
}

void GLSLShader::updateAllFromGLMatrices()
{
	GLdouble modelview[16];
	GLdouble projection[16];
	glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
	glGetDoublev( GL_PROJECTION_MATRIX, projection );

	glm::mat4& model = currentModelView();
	glm::mat4& proj = currentProjection();

	for (unsigned int i=0; i< 4; ++i)
	{
		for (unsigned int j=0; j<4; ++j)
		{
			proj[i][j] = float(projection[4*i+j]);
			model[i][j] = float(modelview[4*i+j]);
		}
	}
	model = currentTransfo() * model;

	currentPMV() = proj * model;
	currentNormalMatrix() = glm::gtx::inverse_transpose::inverseTranspose(model);

	for(std::set< std::pair<void*, GLSLShader*> >::iterator it = m_registeredShaders.begin(); it != m_registeredShaders.end(); ++it)
		it->second->updateMatrices(proj, model, currentPMV(), currentNormalMatrix());
}


} // namespace Utils

} // namespace CGoGN
