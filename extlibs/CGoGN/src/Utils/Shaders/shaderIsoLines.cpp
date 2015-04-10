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
#include <string>
#include "Utils/Shaders/shaderIsoLines.h"

namespace CGoGN
{

namespace Utils
{

#include "shaderIsoLines.vert"
#include "shaderIsoLines.frag"
#include "shaderIsoLines.geom"


ShaderIsoLines::ShaderIsoLines(int maxNbIsoPerTriangle)
{
	m_nameVS = "shaderIsoLines_vs";
	m_nameFS = "shaderIsoLines_fs";
	m_nameGS = "shaderIsoLines_gs";

	std::string glxvert(GLSLShader::defines_gl());
	glxvert.append(vertexShaderText);

	std::string glxgeom = GLSLShader::defines_Geom("triangles", "line_strip", 2*maxNbIsoPerTriangle);
	glxgeom.append(geometryShaderText);

	std::string glxfrag(GLSLShader::defines_gl());
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str(), glxgeom.c_str(), GL_TRIANGLES, GL_LINE_STRIP, 2*maxNbIsoPerTriangle);

	getLocations();

	//Default values

	setColors(Geom::Vec4f(1.0f,0.0f,0.0f,1.0f),Geom::Vec4f(0.0f,1.0f,0.0f,1.0f));
	setDataBound(0.0f,1.0f);
	setNbIso(32);
}

void ShaderIsoLines::getLocations()
{
	bind();
	*m_unif_colorMin = glGetUniformLocation(program_handler(),"colorMin");
	*m_unif_colorMax = glGetUniformLocation(program_handler(),"colorMax");
	*m_unif_vmin = glGetUniformLocation(program_handler(),"vmin");
	*m_unif_vmax = glGetUniformLocation(program_handler(),"vmax");
	*m_unif_vnb = glGetUniformLocation(program_handler(),"vnb");
	unbind();
}

unsigned int ShaderIsoLines::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int ShaderIsoLines::setAttributeData(VBO* vbo)
{
	m_vboData = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexData", vbo);
	unbind();
	return id;
}

void ShaderIsoLines::setColors(const Geom::Vec4f& colorMin, const Geom::Vec4f& colorMax)
{
	m_colorMin = colorMin;
	m_colorMax = colorMax;
	bind();
	glUniform4fv(*m_unif_colorMin,1, m_colorMin.data());
	glUniform4fv(*m_unif_colorMax,1, m_colorMax.data());
	unbind();
}

void ShaderIsoLines::setDataBound(float attMin, float attMax)
{
	m_vmin = attMin;
	m_vmax = attMax;
	bind();
	glUniform1f(*m_unif_vmin, m_vmin);
	glUniform1f(*m_unif_vmax, m_vmax);
	unbind();
}

void ShaderIsoLines::setNbIso(int nb)
{
	m_vnb = nb;
	bind();
	glUniform1i(*m_unif_vnb, m_vnb);
	unbind();
}

/*
void ShaderIsoLines::restoreUniformsAttribs()
{
	bind();

	*m_unif_explode   = glGetUniformLocation(program_handler(),"explode");
	*m_unif_ambiant   = glGetUniformLocation(program_handler(),"ambient");
	*m_unif_lightPos =  glGetUniformLocation(program_handler(),"lightPosition");

	glUniform1f (*m_unif_explode, m_explode);
	glUniform4fv(*m_unif_ambiant,  1, m_ambiant.data());
	glUniform3fv(*m_unif_lightPos, 1, m_light_pos.data());

	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexColor", m_vboPos);

	unbind();
}
*/

} // namespace Utils

} // namespace CGoGN
