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

#include <GL/glew.h>
#include "Utils/Shaders/shaderSimpleTexture.h"


namespace CGoGN
{

namespace Utils
{
#include "shaderSimpleTexture.vert"
#include "shaderSimpleTexture.frag"

//std::string ShaderSimpleTexture::vertexShaderText =
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
//std::string ShaderSimpleTexture::fragmentShaderText =
//		"PRECISON;\n"
//		"VARYING_FRAG vec2 texCoord;\n"
//		"uniform sampler2D textureUnit;\n"
//		"FRAG_OUT_DEF;\n"
//		"void main()\n"
//		"{\n"
//		"	FRAG_OUT=texture2D(textureUnit,texCoord);\n"
//		"}";


ShaderSimpleTexture::ShaderSimpleTexture()
{
	std::string glxvert(*GLSLShader::DEFINES_GL);
	glxvert.append(vertexShaderText);

	std::string glxfrag(*GLSLShader::DEFINES_GL);
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());
	
	m_unif_unit = glGetUniformLocation(this->program_handler(), "textureUnit");
}

void ShaderSimpleTexture::setTextureUnit(GLenum texture_unit)
{
	this->bind();
	int unit = texture_unit - GL_TEXTURE0;
	glUniform1i(*m_unif_unit,unit);
	m_unit = unit;
}

void ShaderSimpleTexture::setTexture(Utils::GTexture* tex)
{
	m_tex_ptr = tex;
}

void ShaderSimpleTexture::activeTexture()
{
	glActiveTexture(GL_TEXTURE0 + m_unit);
	m_tex_ptr->bind();
}

void ShaderSimpleTexture::activeTexture(CGoGNGLuint texId)
{
	glActiveTexture(GL_TEXTURE0 + m_unit);
	glBindTexture(GL_TEXTURE_2D, *texId);
}

unsigned int ShaderSimpleTexture::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	return bindVA_VBO("VertexPosition", vbo);
}

unsigned int ShaderSimpleTexture::setAttributeTexCoord(VBO* vbo)
{
	m_vboTexCoord = vbo;
	return bindVA_VBO("VertexTexCoord", vbo);
}

void ShaderSimpleTexture::restoreUniformsAttribs()
{
	m_unif_unit = glGetUniformLocation(this->program_handler(), "textureUnit");

	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexTexCoord", m_vboTexCoord);
	
	this->bind();
	glUniform1i(*m_unif_unit,m_unit);
	this->unbind();
}

} // namespace Utils

} // namespace CGoGN
