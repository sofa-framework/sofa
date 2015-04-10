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
#define CGoGN_UTILS_DLL_EXPORT 1
#include <GL/glew.h>
#include "Utils/Shaders/shaderCustomTex.h"

#include "shaderCustomTex.vert"
#include "shaderCustomTex.frag"
#include "shaderCustomTex.geom"

//std::string ShaderCustomTex::vertexShaderText =
//		"ATTRIBUTE vec3 VertexPosition;\n"
//		"ATTRIBUTE vec2 VertexTexCoord;\n"
//		"uniform mat4 ModelViewProjectionMatrix;\n"
//		"VARYING_VERT vec2 texCoord;\n"
//		"INVARIANT_POS;\n"
//		"void main ()\n"
//		"{\n"
//		"	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);\n"
//		"	texCoord = VertexTexCoord;\n"
//		"}";
//
//
//std::string ShaderCustomTex::fragmentShaderText =
//		"PRECISION;\n"
//		"VARYING_FRAG vec2 texCoord;\n"
//		"uniform sampler2D textureUnit;\n"
//		"FRAG_OUT_DEF;\n"
//		"void main()\n"
//		"{\n"
//		"	FRAG_OUT=texture2D(textureUnit,texCoord);\n"
//		"}";


ShaderCustomTex::ShaderCustomTex() :
m_col(Geom::Vec4f(1.0f,1.0f,1.0f,1.0f))
{
	m_nameVS = "ShaderCustomTex_vs";
	m_nameFS = "ShaderCustomTex_fs";
	m_nameGS = "ShaderCustomTex_gs";

	std::string glxvert(GLSLShader::defines_gl());
	glxvert.append(vertexShaderText);

	std::string glxgeom = GLSLShader::defines_Geom("triangles", "triangle_strip", 3);
	glxgeom.append(geometryShaderText);

	std::string glxfrag(GLSLShader::defines_gl());
	glxfrag.append(fragmentShaderText);

//	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());
	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str(), glxgeom.c_str(), GL_TRIANGLES, GL_TRIANGLE_STRIP, 3);

	m_unif_unit = glGetUniformLocation(this->program_handler(), "textureUnit");

	setBaseColor(m_col);

	Geom::Matrix44f id;
	id.identity();
	setTransformation(id);

}

void ShaderCustomTex::setTextureUnit(GLenum texture_unit)
{
	this->bind();
	int unit = texture_unit - GL_TEXTURE0;
	glUniform1i(*m_unif_unit,unit);
	m_unit = unit;
}

void ShaderCustomTex::setTexture(Utils::GTexture* tex)
{
	m_tex_ptr = tex;
}

void ShaderCustomTex::setBaseColor(Geom::Vec4f col)
{
	m_col = col;

	bind();
	CGoGNGLuint m_unif_ambiant;
	*m_unif_ambiant  = glGetUniformLocation(program_handler(),"ambient");
	glUniform4fv(*m_unif_ambiant, 1, m_col.data());
	unbind();
}

void ShaderCustomTex::setTransformation(Geom::Matrix44f t)
{
	bind();
	CGoGNGLuint m_transf;
	*m_transf  = glGetUniformLocation(program_handler(),"TransformationMatrix");
	glUniformMatrix4fv(*m_transf, 1, false, &t(0,0));
	unbind();
}

void ShaderCustomTex::activeTexture()
{
	glActiveTexture(GL_TEXTURE0 + m_unit);
	m_tex_ptr->bind();
}

void ShaderCustomTex::activeTexture(CGoGNGLuint texId)
{
	glActiveTexture(GL_TEXTURE0 + m_unit);
	glBindTexture(GL_TEXTURE_2D, *texId);
}

unsigned int ShaderCustomTex::setAttributePosition(Utils::VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int ShaderCustomTex::setAttributeNormal(Utils::VBO* vbo)
{
	m_vboNormal = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexNormal", vbo);
	unbind();
	return id;
}

unsigned int ShaderCustomTex::setAttributeTexCoord(Utils::VBO* vbo)
{
	m_vboTexCoord = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexTexCoord", vbo);
	unbind();
	return id;
}

void ShaderCustomTex::restoreUniformsAttribs()
{
	m_unif_unit = glGetUniformLocation(this->program_handler(), "textureUnit");

	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexTexCoord", m_vboTexCoord);

	this->bind();
	glUniform1i(*m_unif_unit,m_unit);
	this->unbind();
}
