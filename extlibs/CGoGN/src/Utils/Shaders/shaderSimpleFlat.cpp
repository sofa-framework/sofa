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
#include <GL/glew.h>
#include "Utils/Shaders/shaderSimpleFlat.h"

namespace CGoGN
{

namespace Utils
{

#include "shaderSimpleFlat.vert"
#include "shaderSimpleFlat.frag"
#include "shaderSimpleFlatClip.vert"
#include "shaderSimpleFlatClip.frag"



ShaderSimpleFlat::ShaderSimpleFlat(bool withClipping, bool doubleSided):
	m_with_color(false),
	m_ambiant(Geom::Vec4f(0.05f,0.05f,0.1f,0.0f)),
	m_diffuse(Geom::Vec4f(0.1f,1.0f,0.1f,0.0f)),
	m_lightPos(Geom::Vec3f(10.0f,10.0f,1000.0f)),
	m_backColor(0.0f,0.0f,0.0f,0.0f),
	m_vboPos(NULL),
	m_vboColor(NULL),
	m_planeClip(Geom::Vec4f(0.0f,0.0f,0.0f,0.0f))
{
	std::string glxvert(GLSLShader::defines_gl());
	std::string glxfrag(GLSLShader::defines_gl());

	if (withClipping)
	{
		m_nameVS = "ShaderSimpleFlatClip_vs";
		m_nameFS = "ShaderSimpleFlatClip_fs";
		glxvert.append(vertexShaderClipText);
		if (doubleSided)
			glxfrag.append("#define DOUBLE_SIDED\n");
		glxfrag.append(fragmentShaderClipText);
	}
	else
	{
		m_nameVS = "ShaderSimpleFlat_vs";
		m_nameFS = "ShaderSimpleFlat_fs";
		// get choose GL defines (2 or 3)
		// ans compile shaders
		glxvert.append(vertexShaderText);
		if (doubleSided)
			glxfrag.append("#define DOUBLE_SIDED\n");
		glxfrag.append(fragmentShaderText);
	}

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());
	// and get and fill uniforms
	getLocations();
	sendParams();
}

void ShaderSimpleFlat::getLocations()
{
	bind();
	*m_unif_ambiant   = glGetUniformLocation(this->program_handler(), "materialAmbient");
	*m_unif_diffuse   = glGetUniformLocation(this->program_handler(), "materialDiffuse");
	*m_unif_lightPos  = glGetUniformLocation(this->program_handler(), "lightPosition");
	*m_unif_backColor  = glGetUniformLocation(this->program_handler(), "backColor");
	*m_unif_planeClip = glGetUniformLocation(this->program_handler(), "planeClip");
	unbind();
}

void ShaderSimpleFlat::sendParams()
{
	bind();
	glUniform4fv(*m_unif_ambiant,  1, m_ambiant.data());
	glUniform4fv(*m_unif_diffuse,  1, m_diffuse.data());
	glUniform3fv(*m_unif_lightPos, 1, m_lightPos.data());
	glUniform4fv(*m_unif_backColor,  1, m_backColor.data());
	if (*m_unif_planeClip > 0)
		glUniform4fv(*m_unif_planeClip, 1, m_planeClip.data());
	unbind();
}

void ShaderSimpleFlat::setAmbiant(const Geom::Vec4f& ambiant)
{
	bind();
	glUniform4fv(*m_unif_ambiant,1, ambiant.data());
	m_ambiant = ambiant;
	unbind();
}

void ShaderSimpleFlat::setDiffuse(const Geom::Vec4f& diffuse)
{
	bind();
	glUniform4fv(*m_unif_diffuse,1, diffuse.data());
	m_diffuse = diffuse;
	unbind();
}


void ShaderSimpleFlat::setLightPosition(const Geom::Vec3f& lightPos)
{
	bind();
	glUniform3fv(*m_unif_lightPos,1,lightPos.data());
	m_lightPos = lightPos;
	unbind();
}


void ShaderSimpleFlat::setBackColor(const Geom::Vec4f& back)
{
	bind();
	glUniform4fv(*m_unif_backColor,1, back.data());
	m_backColor = back;
	unbind();
}


void ShaderSimpleFlat::setParams(const Geom::Vec4f& ambiant, const Geom::Vec4f& diffuse, const Geom::Vec3f& lightPos)
{
	m_ambiant = ambiant;
	m_diffuse = diffuse;
	m_lightPos = lightPos;
	sendParams();
}

unsigned int ShaderSimpleFlat::setAttributeColor(VBO* vbo)
{
	m_vboColor = vbo;
	if (!m_with_color)
	{
		m_with_color=true;
		// set the define and recompile shader
		std::string gl3vert(GLSLShader::defines_gl());
		gl3vert.append("#define WITH_COLOR 1\n");
		gl3vert.append(vertexShaderText);
		std::string gl3frag(GLSLShader::defines_gl());
		gl3frag.append("#define WITH_COLOR 1\n");
		gl3frag.append(fragmentShaderText);
		loadShadersFromMemory(gl3vert.c_str(), gl3frag.c_str());

		// and treat uniforms
		getLocations();
		sendParams();
	}
	// bind th VA with WBO
	bind();
	unsigned int id = bindVA_VBO("VertexColor", vbo);
	unbind();
	return id;
}

void ShaderSimpleFlat::unsetAttributeColor()
{
	m_vboColor = NULL;
	if (m_with_color)
	{
		m_with_color = false;
		// unbind the VA
		bind();
		unbindVA("VertexColor");
		unbind();
		// recompile shader
		std::string gl3vert(GLSLShader::defines_gl());
		gl3vert.append(vertexShaderText);
		std::string gl3frag(GLSLShader::defines_gl());
		gl3frag.append(fragmentShaderText);
		loadShadersFromMemory(gl3vert.c_str(), gl3frag.c_str());
		// and treat uniforms
		getLocations();
		sendParams();
	}
}

void ShaderSimpleFlat::restoreUniformsAttribs()
{
	getLocations();
	sendParams();

	bind();
	bindVA_VBO("VertexPosition", m_vboPos);
	if (m_vboColor)
		bindVA_VBO("VertexColor", m_vboColor);

	unbind();
}

unsigned int ShaderSimpleFlat::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

void ShaderSimpleFlat::setClippingPlane(const Geom::Vec4f& plane)
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
