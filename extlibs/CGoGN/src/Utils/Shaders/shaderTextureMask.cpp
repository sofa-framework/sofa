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
//#ifdef CGOGN_WITH_QT

#include "Utils/Shaders/shaderTextureMask.h"


namespace CGoGN
{

namespace Utils
{

#include "shaderTextureMask.vert"
#include "shaderTextureMask.frag"



ShaderTextureMask::ShaderTextureMask()
{
	std::string glxvert(*GLSLShader::DEFINES_GL);
	glxvert.append(vertexShaderText);

	std::string glxfrag(*GLSLShader::DEFINES_GL);
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());

	bind();
	m_unif_unit = glGetUniformLocation(this->program_handler(), "textureUnit");
	m_unif_unitMask = glGetUniformLocation(this->program_handler(), "textureUnitMask");
	unbind();
}

void ShaderTextureMask::setTextureUnits(GLenum texture_unit, GLenum texture_unitMask)
{
	bind();
	int unit = texture_unit - GL_TEXTURE0;
	glUniform1i(*m_unif_unit,unit);
	m_unit = unit;
	unit = texture_unitMask - GL_TEXTURE0;
	glUniform1i(*m_unif_unitMask,unit);
	m_unitMask = unit;
	unbind();
}

void ShaderTextureMask::setTextures(Utils::GTexture* tex, Utils::GTexture* texMask)
{
	m_tex_ptr = tex;
	m_texMask_ptr = texMask;
}

void ShaderTextureMask::activeTextures()
{
	glActiveTexture(GL_TEXTURE0+m_unit);
	m_tex_ptr->bind();
	glActiveTexture(GL_TEXTURE0+m_unitMask);
	m_texMask_ptr->bind();
}

unsigned int ShaderTextureMask::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int ShaderTextureMask::setAttributeTexCoord(VBO* vbo)
{
	m_vboTexCoord = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexTexCoord", vbo);
	unbind();
	return id;
}

void ShaderTextureMask::restoreUniformsAttribs()
{
	bind();
	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexTexCoord", m_vboTexCoord);
	glUniform1i(*m_unif_unit,m_unit);
	glUniform1i(*m_unif_unitMask,m_unitMask);
	unbind();
}

} // namespace Utils

} // namespace CGoGN

//#else
//#pragma message(__FILE__ " not compiled because of mising Qt")
//#endif
