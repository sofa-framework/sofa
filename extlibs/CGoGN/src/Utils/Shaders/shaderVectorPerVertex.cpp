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
#include "Utils/Shaders/shaderVectorPerVertex.h"

namespace CGoGN
{

namespace Utils
{

#include "shaderVectorPerVertex.vert"
#include "shaderVectorPerVertex.geom"
#include "shaderVectorPerVertex.frag"


ShaderVectorPerVertex::ShaderVectorPerVertex() :
	m_scale(1.0f),
	m_color	(1.0f, 0.0f, 0.0f, 0.0f),
	m_planeClip(0.0f,0.0f,0.0f,0.0f)
{
	m_nameVS = "ShaderVectorPerVertex_vs";
	m_nameFS = "ShaderVectorPerVertex_fs";
	m_nameGS = "ShaderVectorPerVertex_gs";

	std::string glxvert(GLSLShader::defines_gl());
	glxvert.append(vertexShaderText);

	std::string glxgeom = GLSLShader::defines_Geom("points", "line_strip", 4);
	glxgeom.append(geometryShaderText);

	std::string glxfrag(GLSLShader::defines_gl());
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str(), glxgeom.c_str(), GL_POINTS, GL_LINE_STRIP,2);

	// get and fill uniforms
	getLocations();
	sendParams();
}

void ShaderVectorPerVertex::getLocations()
{
	bind();
	*m_uniform_scale = glGetUniformLocation(this->program_handler(), "vectorScale");
	*m_uniform_color = glGetUniformLocation(this->program_handler(), "vectorColor");
	*m_unif_planeClip = glGetUniformLocation(this->program_handler(), "planeClip");
	unbind();
}

void ShaderVectorPerVertex::sendParams()
{
	bind();
	glUniform1f(*m_uniform_scale, m_scale);
	glUniform4fv(*m_uniform_color, 1, m_color.data());
	if (*m_unif_planeClip > 0)
		glUniform4fv(*m_unif_planeClip, 1, m_planeClip.data());

	unbind();
}

void ShaderVectorPerVertex::setScale(float scale)
{
	bind();
	glUniform1f(*m_uniform_scale, scale);
	m_scale = scale;
	unbind();
}

void ShaderVectorPerVertex::setColor(const Geom::Vec4f& color)
{
	bind();
	glUniform4fv(*m_uniform_color, 1, color.data());
	m_color = color;
	unbind();
}

unsigned int ShaderVectorPerVertex::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int ShaderVectorPerVertex::setAttributeVector(VBO* vbo)
{
	m_vboVec = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexVector", vbo);
	unbind();
	return id;
}

void ShaderVectorPerVertex::restoreUniformsAttribs()
{
	getLocations();
	sendParams();

	bind();
	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexVector", m_vboVec);
	unbind();
}

void ShaderVectorPerVertex::setClippingPlane(const Geom::Vec4f& plane)
{
	if (*m_unif_planeClip > 0)
	{
		m_planeClip = plane;
		bind();
		glUniform4fv(*m_unif_planeClip, 1, plane.data());
		unbind();
	}
}


} // namespace Utils

} // namespace CGoGN
