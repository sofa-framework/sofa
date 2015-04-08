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
#include "Utils/pointSprite.h"

namespace CGoGN
{

namespace Utils
{

#include "pointSprite.vert"
#include "pointSprite.frag"
#include "pointSprite.geom"

PointSprite::PointSprite(bool withColorPerVertex, bool withPlane) :
	colorPerVertex(withColorPerVertex),
	plane(withPlane),
	m_size(1.0f),
	m_color(Geom::Vec4f(0.0f, 0.0f, 1.0f, 1.0f)),
	m_lightPos(Geom::Vec3f(100.0f, 100.0f, 100.0f)),
	m_ambiant(Geom::Vec3f(0.1f, 0.1f, 0.1f)),
	m_eyePos(Geom::Vec3f(0.0f, 0.0f, 0.0f)),
    m_planeClip(Geom::Vec4f(0.0f,0.0f,0.0f,0.0f))
{
	std::string glxvert(GLSLShader::defines_gl());
	if (withColorPerVertex)
		glxvert.append("#define WITH_COLOR_PER_VERTEX 1\n");
	glxvert.append(vertexShaderText);

	std::string glxgeom = GLSLShader::defines_Geom("points", "triangle_strip", 4);
	if (withColorPerVertex)
		glxgeom.append("#define WITH_COLOR_PER_VERTEX 1\n");
	if (withPlane)
		glxgeom.append("#define WITH_PLANE 1\n");
	glxgeom.append(geometryShaderText);

	std::string glxfrag(GLSLShader::defines_gl());
	if (withColorPerVertex)
		glxfrag.append("#define WITH_COLOR_PER_VERTEX 1\n");
	if (withPlane)
		glxfrag.append("#define WITH_PLANE 1\n");
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str(), glxgeom.c_str(), GL_POINTS, GL_TRIANGLE_STRIP, 4);

	// get and fill uniforms
	getLocations();
	sendParams();
}

void PointSprite::getLocations()
{
	bind();
	*m_uniform_size = glGetUniformLocation(program_handler(),"size");
	if (!colorPerVertex)
		*m_uniform_color = glGetUniformLocation(program_handler(),"colorsprite");
	*m_uniform_ambiant = glGetUniformLocation(program_handler(),"ambiant");
	*m_uniform_lightPos = glGetUniformLocation(program_handler(),"lightPos");
	if (plane)
		*m_uniform_eyePos = glGetUniformLocation(program_handler(),"eyePos");
	*m_unif_planeClip = glGetUniformLocation(this->program_handler(), "planeClip");
	unbind();
}

void PointSprite::sendParams()
{
	bind();
	glUniform1f(*m_uniform_size, m_size);
	if (!colorPerVertex)
		glUniform4fv(*m_uniform_color, 1, m_color.data());
	glUniform3fv(*m_uniform_ambiant, 1, m_ambiant.data());
	glUniform3fv(*m_uniform_lightPos, 1, m_lightPos.data());
	if (plane)
		glUniform3fv(*m_uniform_eyePos, 1, m_eyePos.data());
	if (*m_unif_planeClip > 0)
		glUniform4fv(*m_unif_planeClip, 1, m_planeClip.data());
	unbind();
}

unsigned int PointSprite::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int PointSprite::setAttributeColor(VBO* vbo)
{
	if (colorPerVertex)
	{
		m_vboColor = vbo;
		bind();
		unsigned int id = bindVA_VBO("VertexColor", vbo);
		unbind();
		return id;
	}
	return 0;
}

void PointSprite::setSize(float size)
{
	m_size = size;
	bind();
	glUniform1f(*m_uniform_size, size);
	unbind();
}

void PointSprite::setColor(const Geom::Vec4f& color)
{
	if (!colorPerVertex)
	{
		m_color = color;
		bind();
		glUniform4fv(*m_uniform_color, 1, color.data());
		unbind();
	}
}

void PointSprite::setLightPosition(const Geom::Vec3f& pos)
{
	m_lightPos = pos;
	bind();
	glUniform3fv(*m_uniform_lightPos, 1, pos.data());
	unbind();
}

void PointSprite::setAmbiantColor(const Geom::Vec3f& amb)
{
	m_ambiant = amb;
	bind();
	glUniform3fv(*m_uniform_ambiant, 1, amb.data());
	unbind();
}

void PointSprite::setEyePosition(const Geom::Vec3f& ep)
{
	if (plane)
	{
		m_eyePos = ep;
		bind();
		glUniform3fv(*m_uniform_eyePos, 1, ep.data());
		unbind();
	}
}

void PointSprite::setClippingPlane(const Geom::Vec4f& plane)
{
	m_planeClip = plane;
	bind();
	glUniform4fv(*m_unif_planeClip, 1, plane.data());
	unbind();
}


void PointSprite::restoreUniformsAttribs()
{
	getLocations();
	sendParams();

	bind();
	bindVA_VBO("VertexPosition", m_vboPos);
	if (colorPerVertex)
		bindVA_VBO("VertexColor", m_vboColor);
	unbind();
}

} // namespace Utils

} // namespace CGoGN
