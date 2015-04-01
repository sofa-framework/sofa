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
#include "Utils/Shaders/shaderFlatColor.h"

namespace CGoGN
{

namespace Utils
{
#include "shaderFlatColor.vert"
#include "shaderFlatColor.frag"
#include "shaderFlatColor.geom"


ShaderFlatColor::ShaderFlatColor(bool averageColor)
{
	m_nameVS = "shaderFlatColor_vs";
	m_nameFS = "shaderFlatColor_fs";
	m_nameGS = "shaderFlatColor_gs";

	std::string glxvert(GLSLShader::defines_gl());
	glxvert.append(vertexShaderText);

	std::string glxgeom = GLSLShader::defines_Geom("triangles", "triangle_strip", 3);
	if (averageColor)
		glxgeom.append("#define AVERAGE_COLOR 1\n");
	glxgeom.append(geometryShaderText);

	std::string glxfrag(GLSLShader::defines_gl());
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str(), glxgeom.c_str(), GL_TRIANGLES, GL_TRIANGLE_STRIP,3);

	bind();
	getLocations();
	unbind();

	//Default values
	m_explode = 1.0f;
	m_ambiant = Geom::Vec4f(0.05f, 0.05f, 0.1f, 0.0f);
	m_light_pos = Geom::Vec3f(10.0f, 10.0f, 1000.0f);
	setParams(m_explode, m_ambiant, m_light_pos);
}

// precond : is binded
void ShaderFlatColor::getLocations()
{
	*m_unif_explode  = glGetUniformLocation(program_handler(),"explode");
	*m_unif_ambiant  = glGetUniformLocation(program_handler(),"ambient");
	*m_unif_lightPos = glGetUniformLocation(program_handler(),"lightPosition");
}

unsigned int ShaderFlatColor::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int ShaderFlatColor::setAttributeColor(VBO* vbo)
{
	m_vboColor = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexColor", vbo);
	unbind();
	return id;
}

void ShaderFlatColor::setParams(float expl, const Geom::Vec4f& ambiant, const Geom::Vec3f& lightPos)
{
	m_explode = expl;
	m_ambiant = ambiant;
	m_light_pos = lightPos;

	bind();

	glUniform1f(*m_unif_explode, expl);
	glUniform4fv(*m_unif_ambiant, 1, ambiant.data());
	glUniform3fv(*m_unif_lightPos, 1, lightPos.data());

	unbind();
}

void ShaderFlatColor::setExplode(float explode)
{
	m_explode = explode;
	bind();
	glUniform1f(*m_unif_explode, explode);
	unbind();
}

void ShaderFlatColor::setAmbiant(const Geom::Vec4f& ambiant)
{
	m_ambiant = ambiant;
	bind();
	glUniform4fv(*m_unif_ambiant,1, ambiant.data());
	unbind();
}

void ShaderFlatColor::setLightPosition(const Geom::Vec3f& lp)
{
	m_light_pos = lp;
	bind();
	glUniform3fv(*m_unif_lightPos,1,lp.data());
	unbind();
}

void ShaderFlatColor::restoreUniformsAttribs()
{
	*m_unif_explode   = glGetUniformLocation(program_handler(),"explode");
	*m_unif_ambiant   = glGetUniformLocation(program_handler(),"ambient");
	*m_unif_lightPos =  glGetUniformLocation(program_handler(),"lightPosition");

	bind();
	glUniform1f (*m_unif_explode, m_explode);
	glUniform4fv(*m_unif_ambiant,  1, m_ambiant.data());
	glUniform3fv(*m_unif_lightPos, 1, m_light_pos.data());

	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexColor", m_vboPos);
	unbind();
}

} // namespace Utils

} // namespace CGoGN
