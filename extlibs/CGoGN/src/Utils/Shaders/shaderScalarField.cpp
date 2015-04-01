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
#include "Utils/Shaders/shaderScalarField.h"

namespace CGoGN
{

namespace Utils
{

#include "shaderScalarField.vert"
#include "shaderScalarField.frag"

ShaderScalarField::ShaderScalarField() :
	m_minValue(0.0f),
	m_maxValue(0.0f),
	m_expansion(0)
{
	std::string glxvert(GLSLShader::defines_gl());
	glxvert.append(vertexShaderText);

	std::string glxfrag(GLSLShader::defines_gl());
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());

	// get and fill uniforms
	getLocations();
	sendParams();
}

void ShaderScalarField::getLocations()
{
	bind();
	*m_uniform_minValue = glGetUniformLocation(this->program_handler(), "minValue");
	*m_uniform_maxValue = glGetUniformLocation(this->program_handler(), "maxValue");
	*m_uniform_colorMap = glGetUniformLocation(this->program_handler(), "colorMap");
	*m_uniform_expansion = glGetUniformLocation(this->program_handler(), "expansion");
	unbind();
}

void ShaderScalarField::sendParams()
{
	bind();
	glUniform1f(*m_uniform_minValue, m_minValue);
	glUniform1f(*m_uniform_maxValue, m_maxValue);
	glUniform1i(*m_uniform_colorMap, m_colorMap);
	glUniform1i(*m_uniform_expansion, m_expansion);
	unbind();
}

unsigned int ShaderScalarField::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int ShaderScalarField::setAttributeScalar(VBO* vbo)
{
	m_vboScal = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexScalar", vbo);
	unbind();
	return id;
}

void ShaderScalarField::setMinValue(float f)
{
	m_minValue = f;
	bind();
	glUniform1f(*m_uniform_minValue, f);
	unbind();
}

void ShaderScalarField::setMaxValue(float f)
{
	m_maxValue = f;
	bind();
	glUniform1f(*m_uniform_maxValue, f);
	unbind();
}

void ShaderScalarField::setColorMap(int i)
{
	m_colorMap = i;
	bind();
	glUniform1i(*m_uniform_colorMap, i);
	unbind();
}

void ShaderScalarField::setExpansion(int i)
{
	m_expansion = i;
	bind();
	glUniform1i(*m_uniform_expansion, i);
	unbind();
}

void ShaderScalarField::restoreUniformsAttribs()
{
	getLocations();
	sendParams();

	bind();
	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexScalar", m_vboScal);
	unbind();
}

} // namespace Utils

} // namespace CGoGN
