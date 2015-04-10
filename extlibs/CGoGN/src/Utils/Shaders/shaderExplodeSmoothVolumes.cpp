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
#include "Utils/Shaders/shaderExplodeSmoothVolumes.h"

namespace CGoGN
{

namespace Utils
{

#include "shaderExplodeSmoothVolumes.vert"
#include "shaderExplodeSmoothVolumes.frag"
#include "shaderExplodeSmoothVolumes.geom"


ShaderExplodeSmoothVolumes::ShaderExplodeSmoothVolumes(bool withColorPerFace, bool withExplodeFace):
	m_wcpf(withColorPerFace),
	m_wef(withExplodeFace)
{
	m_nameVS = "ShaderExplodeSmoothVolumes_vs";
	m_nameFS = "ShaderExplodeSmoothVolumes_fs";
	m_nameGS = "ShaderExplodeSmoothVolumes_gs";

	std::string glxvert(GLSLShader::defines_gl());
	glxvert.append(vertexShaderText);

	std::string glxgeom;
	glxgeom.append(GLSLShader::defines_Geom("lines_adjacency", "triangle_strip", 3));

	if (withColorPerFace)
		glxgeom.append("#define WITH_COLORPF 1\n");
	if (withExplodeFace)
		glxgeom.append("#define WITH_EXPLODE_FACE 1\n");
	glxgeom.append(geometryShaderText);

	std::string glxfrag(GLSLShader::defines_gl());
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str(), glxgeom.c_str(), GL_LINES_ADJACENCY_EXT , GL_TRIANGLE_STRIP,4);

	getLocations();

	//Default values
	m_explodeV = 0.9f;
	m_explodeF = 0.9f;
	m_ambiant = Geom::Vec4f(0.05f, 0.05f, 0.1f, 0.0f);
	m_light_pos = Geom::Vec3f(10.0f, 10.0f, 1000.0f);
	m_plane   = Geom::Vec4f(0.0f, 0.0f, 1000.f, 1000000000000000000000000000.0f);

	setParams(m_explodeV, m_explodeF, m_ambiant, m_light_pos, m_plane);
}

void ShaderExplodeSmoothVolumes::getLocations()
{
	bind();
	*m_unif_explodeV  = glGetUniformLocation(program_handler(),"explodeV");
	*m_unif_explodeF  = glGetUniformLocation(program_handler(),"explodeF");
	*m_unif_ambiant  = glGetUniformLocation(program_handler(),"ambient");
	*m_unif_lightPos = glGetUniformLocation(program_handler(),"lightPosition");
	*m_unif_plane   = glGetUniformLocation(program_handler(),"plane");
	unbind();
}

unsigned int ShaderExplodeSmoothVolumes::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int ShaderExplodeSmoothVolumes::setAttributeColor(VBO* vbo)
{
	m_vboColors = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexColor", vbo);
	unbind();
	return id;
}

unsigned int ShaderExplodeSmoothVolumes::setAttributeNormal(VBO* vbo)
{
	m_vboNormals = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexNormal", vbo);
	unbind();
	return id;
}


void ShaderExplodeSmoothVolumes::setParams(float explV, float explF, const Geom::Vec4f& ambiant, const Geom::Vec3f& lightPos, const Geom::Vec4f& plane)
{
	bind();
	m_explodeV = explV;
	glUniform1f(*m_unif_explodeV, explV);
	m_explodeF = explF;
	glUniform1f(*m_unif_explodeF, explF);
	m_ambiant = ambiant;
	glUniform4fv(*m_unif_ambiant, 1, ambiant.data());
	m_light_pos = lightPos;
	glUniform3fv(*m_unif_lightPos, 1, lightPos.data());

	m_plane = plane;
	glUniform4fv(*m_unif_plane,    1, m_plane.data());

	unbind();
}

void ShaderExplodeSmoothVolumes::setExplodeVolumes(float explode)
{
	m_explodeV = explode;
	bind();
	glUniform1f(*m_unif_explodeV, explode);
	unbind();
}

void ShaderExplodeSmoothVolumes::setExplodeFaces(float explode)
{
	m_explodeF = explode;
	bind();
	glUniform1f(*m_unif_explodeF, explode);
	unbind();
}

void ShaderExplodeSmoothVolumes::setAmbiant(const Geom::Vec4f& ambiant)
{
	m_ambiant = ambiant;
	bind();
	glUniform4fv(*m_unif_ambiant,1, ambiant.data());
	unbind();
}



void ShaderExplodeSmoothVolumes::setLightPosition(const Geom::Vec3f& lp)
{
	m_light_pos = lp;
	bind();
	glUniform3fv(*m_unif_lightPos,1,lp.data());
	unbind();
}

void ShaderExplodeSmoothVolumes::setClippingPlane(const Geom::Vec4f& plane)
{
	m_plane = plane;
	bind();
	glUniform4fv(*m_unif_plane,1, plane.data());
	unbind();
}

void ShaderExplodeSmoothVolumes::restoreUniformsAttribs()
{
	bind();

	*m_unif_explodeV   = glGetUniformLocation(program_handler(),"explodeV");
	glUniform1f (*m_unif_explodeV, m_explodeV);

	*m_unif_explodeF   = glGetUniformLocation(program_handler(),"explodeF");
	glUniform1f (*m_unif_explodeF, m_explodeF);

	*m_unif_ambiant   = glGetUniformLocation(program_handler(),"ambient");
	glUniform4fv(*m_unif_ambiant,  1, m_ambiant.data());

	*m_unif_lightPos =  glGetUniformLocation(program_handler(),"lightPosition");
	glUniform3fv(*m_unif_lightPos, 1, m_light_pos.data());

	*m_unif_plane   = glGetUniformLocation(program_handler(),"plane");
	glUniform4fv(*m_unif_plane,    1, m_plane.data());

	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexColor", m_vboColors);
	bindVA_VBO("VertexNormal", m_vboNormals);

	unbind();
}

} // namespace Utils

} // namespace CGoGN
