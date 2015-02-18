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

#include <string.h>
#include "Utils/Shaders/shaderFlat.h"

namespace CGoGN
{

namespace Utils
{

#include "shaderFlat.vert"
#include "shaderFlat.frag"
#include "shaderFlat.geom"


ShaderFlat::ShaderFlat()
{
	m_nameVS = "ShaderFlat_vs";
	m_nameFS = "ShaderFlat_fs";
	m_nameGS = "ShaderFlat_gs";

	std::string glxvert(*GLSLShader::DEFINES_GL);
	glxvert.append(vertexShaderText);

	std::string glxgeom = GLSLShader::defines_Geom("triangles", "triangle_strip", 3);
	glxgeom.append(geometryShaderText);

	std::string glxfrag(*GLSLShader::DEFINES_GL);
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str(), glxgeom.c_str(), GL_TRIANGLES, GL_TRIANGLE_STRIP, 3);

	bind();
	getLocations();
	unbind();

	//Default values
	m_explode = 1.0f;
	m_ambiant = Geom::Vec4f(0.05f, 0.05f, 0.1f, 0.0f);
	m_diffuse = Geom::Vec4f(0.1f, 1.0f, 0.1f, 0.0f);
	m_diffuseBack = m_diffuse;
	m_light_pos = Geom::Vec3f(10.0f, 10.0f, 1000.0f);

	setParams(m_explode, m_ambiant, m_diffuse, m_diffuseBack ,m_light_pos);
}

void ShaderFlat::getLocations()
{
	*m_unif_explode  = glGetUniformLocation(program_handler(), "explode");
	*m_unif_ambiant  = glGetUniformLocation(program_handler(), "ambient");
	*m_unif_diffuse  = glGetUniformLocation(program_handler(), "diffuse");
	*m_unif_diffuseback  = glGetUniformLocation(program_handler(), "diffuseBack");
	*m_unif_lightPos = glGetUniformLocation(program_handler(), "lightPosition");
}

unsigned int ShaderFlat::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

void ShaderFlat::setParams(float expl, const Geom::Vec4f& ambiant, const Geom::Vec4f& diffuse, const Geom::Vec4f& diffuseBack, const Geom::Vec3f& lightPos)
{
	m_explode = expl;
	m_ambiant = ambiant;
	m_diffuse = diffuse;
	m_light_pos = lightPos;

	bind();

	glUniform1f(*m_unif_explode, expl);
	glUniform4fv(*m_unif_ambiant, 1, ambiant.data());
	glUniform4fv(*m_unif_diffuse, 1, diffuse.data());
	glUniform4fv(*m_unif_diffuseback, 1, diffuseBack.data());
	glUniform3fv(*m_unif_lightPos, 1, lightPos.data());

	unbind();
}

void ShaderFlat::setExplode(float explode)
{
	m_explode = explode;
	bind();
	glUniform1f(*m_unif_explode, explode);
	unbind();
}

void ShaderFlat::setAmbiant(const Geom::Vec4f& ambiant)
{
	m_ambiant = ambiant;
	bind();
	glUniform4fv(*m_unif_ambiant,1, ambiant.data());
	unbind();
}

void ShaderFlat::setDiffuse(const Geom::Vec4f& diffuse)
{
	m_diffuse = diffuse;
	bind();
	glUniform4fv(*m_unif_diffuse,1, diffuse.data());
	unbind();
}

void ShaderFlat::setDiffuseBack(const Geom::Vec4f& diffuseb)
{
	m_diffuseBack = diffuseb;
	bind();
	glUniform4fv(*m_unif_diffuseback,1, diffuseb.data());
	unbind();
}

void ShaderFlat::setLightPosition(const Geom::Vec3f& lp)
{
	m_light_pos = lp;
	bind();
	glUniform3fv(*m_unif_lightPos,1,lp.data());
	unbind();
}

void ShaderFlat::restoreUniformsAttribs()
{
	*m_unif_explode     = glGetUniformLocation(program_handler(),"explode");
	*m_unif_ambiant     = glGetUniformLocation(program_handler(),"ambient");
	*m_unif_diffuse     = glGetUniformLocation(program_handler(),"diffuse");
	*m_unif_diffuseback = glGetUniformLocation(program_handler(),"diffuseBack");
	*m_unif_lightPos    =  glGetUniformLocation(program_handler(),"lightPosition");

	bind();

	glUniform1f (*m_unif_explode, m_explode);
	glUniform4fv(*m_unif_ambiant,  1, m_ambiant.data());
	glUniform4fv(*m_unif_diffuse,  1, m_diffuse.data());
	glUniform4fv(*m_unif_diffuseback,  1, m_diffuseBack.data());
	glUniform3fv(*m_unif_lightPos, 1, m_light_pos.data());

	bindVA_VBO("VertexPosition", m_vboPos);

	unbind();
}

} // namespace Utils

} // namespace CGoGN
