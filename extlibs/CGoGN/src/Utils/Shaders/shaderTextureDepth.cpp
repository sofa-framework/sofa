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
#include "Utils/Shaders/shaderTextureDepth.h"


namespace CGoGN
{

namespace Utils
{
#include "shaderTextureDepth.vert"
#include "shaderTextureDepth.frag"


ShaderTextureDepth::ShaderTextureDepth()
{
	m_nameVS = "ShaderTextureDepth_vs";
	m_nameFS = "ShaderTextureDepth_fs";

	std::string glxvert(*GLSLShader::DEFINES_GL);
	glxvert.append(vertexShaderText);

	std::string glxfrag(*GLSLShader::DEFINES_GL);

	std::stringstream ss;
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());

	m_unif_unit = glGetUniformLocation(this->program_handler(), "textureUnit");
	m_unif_depthUnit = glGetUniformLocation(this->program_handler(), "textureDepthUnit");
}

void ShaderTextureDepth::setTextureUnit(GLenum texture_unit)
{
	this->bind();
	int unit = texture_unit - GL_TEXTURE0;
	glUniform1i(*m_unif_unit,unit);
	m_unit = unit;
}

void ShaderTextureDepth::setDepthTextureUnit(GLenum texture_unit)
{
	this->bind();
	int unit = texture_unit - GL_TEXTURE0;
	glUniform1i(*m_unif_depthUnit,unit);
	m_depthUnit = unit;
}

void ShaderTextureDepth::activeTexture(CGoGNGLuint texId)
{
	glActiveTexture(GL_TEXTURE0 + m_unit);
	glBindTexture(GL_TEXTURE_2D, *texId);
}
void ShaderTextureDepth::activeDepthTexture(CGoGNGLuint texId)
{
	glActiveTexture(GL_TEXTURE0 + m_depthUnit);
	glBindTexture(GL_TEXTURE_2D, *texId);
}

unsigned int ShaderTextureDepth::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	return bindVA_VBO("VertexPosition", vbo);
}

unsigned int ShaderTextureDepth::setAttributeTexCoord(VBO* vbo)
{
	m_vboTexCoord = vbo;
	return bindVA_VBO("VertexTexCoord", vbo);
}

void ShaderTextureDepth::restoreUniformsAttribs()
{
	m_unif_unit = glGetUniformLocation(this->program_handler(), "textureUnit");
	m_unif_depthUnit = glGetUniformLocation(this->program_handler(), "textureDepthUnit");
	
	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexTexCoord", m_vboTexCoord);
	
	this->bind();
	glUniform1i(*m_unif_unit,m_unit);
	glUniform1i(*m_unif_depthUnit,m_depthUnit);
	this->unbind();
}

} // namespace Utils

} // namespace CGoGN

