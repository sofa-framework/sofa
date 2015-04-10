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
#include <string.h>
#include "Utils/Shaders/shaderExplodeVolumesLines.h"

namespace CGoGN
{

namespace Utils
{

#include "shaderExplodeVolumesLines.vert"
#include "shaderExplodeVolumesLines.frag"
#include "shaderExplodeVolumesLines.geom"


ShaderExplodeVolumesLines::ShaderExplodeVolumesLines()
{
	m_nameVS = "ShaderExplodeVolumesLines_vs";
	m_nameFS = "ShaderExplodeVolumesLines_fs";
	m_nameGS = "ShaderExplodeVolumesLines_gs";

	std::string glxvert(GLSLShader::defines_gl());
	glxvert.append(vertexShaderText);

	std::string glxgeom(GLSLShader::defines_Geom("triangles", "line_strip", 4));
	glxgeom.append(geometryShaderText);

	std::string glxfrag(GLSLShader::defines_gl());
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str(), glxgeom.c_str(), GL_TRIANGLES , GL_LINE_STRIP,4);

	getLocations();

	//Default values
	m_explodeV = 0.9f;
	m_color = Geom::Vec4f(0.05f, 0.05f, 0.05f, 0.0f);
	m_plane   = Geom::Vec4f(0.0f, 0.0f, 1000.f, 1000000000000000000000000000.0f);
	setParams(m_explodeV, m_color, m_plane);
}

void ShaderExplodeVolumesLines::getLocations()
{
	bind();
	*m_unif_explodeV  = glGetUniformLocation(program_handler(),"explodeV");
	*m_unif_color  = glGetUniformLocation(program_handler(),"color");
	*m_unif_plane   = glGetUniformLocation(program_handler(),"plane");
	unbind();
}

unsigned int ShaderExplodeVolumesLines::setAttributePosition(VBO* vbo)
{
	bind();
	m_vboPos = vbo;
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

void ShaderExplodeVolumesLines::setParams(float explV, const Geom::Vec4f& color, const Geom::Vec4f& plane)
{
	m_explodeV = explV;
	m_color = color;
	m_plane = plane;

	bind();

	glUniform1f(*m_unif_explodeV, explV);
	glUniform4fv(*m_unif_color, 1, color.data());
	glUniform4fv(*m_unif_plane,    1, m_plane.data());

	unbind();
}

void ShaderExplodeVolumesLines::setExplodeVolumes(float explode)
{
	m_explodeV = explode;
	bind();
	glUniform1f(*m_unif_explodeV, explode);
	unbind();
}


void ShaderExplodeVolumesLines::setColor(const Geom::Vec4f& color)
{
	m_color = color;
	bind();
	glUniform4fv(*m_unif_color,1, color.data());
	unbind();
}

const Geom::Vec4f& ShaderExplodeVolumesLines::getColor() const
{
	return m_color;
}

void ShaderExplodeVolumesLines::setClippingPlane(const Geom::Vec4f& plane)
{
	m_plane = plane;
	bind();
	glUniform4fv(*m_unif_plane,1, plane.data());
	unbind();
}

void ShaderExplodeVolumesLines::restoreUniformsAttribs()
{
	*m_unif_explodeV   = glGetUniformLocation(program_handler(),"explodeV");
	*m_unif_color   = glGetUniformLocation(program_handler(),"color");
	*m_unif_plane   = glGetUniformLocation(program_handler(),"plane");

	bind();

	glUniform1f (*m_unif_explodeV, m_explodeV);
	glUniform4fv(*m_unif_color,  1, m_color.data());
	glUniform4fv(*m_unif_plane,    1, m_plane.data());

	bindVA_VBO("VertexPosition", m_vboPos);

	unbind();
}

} // namespace Utils

} // namespace CGoGN
