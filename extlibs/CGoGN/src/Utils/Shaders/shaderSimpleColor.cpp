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
#include "Utils/Shaders/shaderSimpleColor.h"

namespace CGoGN
{

namespace Utils
{

#include "shaderSimpleColor.vert"
#include "shaderSimpleColor.frag"
#include "shaderSimpleColorClip.vert"
#include "shaderSimpleColorClip.frag"


ShaderSimpleColor::ShaderSimpleColor(bool withClipping, bool black_is_transparent)
{
	if (withClipping)
	{
		m_nameVS = "ShaderSimpleColorClip_vs";
		m_nameFS = "ShaderSimpleColorClip_fs";
		m_nameGS = "";

		// chose GL defines (2 or 3)
		// and compile shaders
		std::string glxvert(GLSLShader::defines_gl());
		glxvert.append(vertexShaderClipText);

		std::string glxfrag(GLSLShader::defines_gl());
		if (black_is_transparent)
			glxfrag.append("#define BLACK_TRANSPARENCY 1\n");
		glxfrag.append(fragmentShaderClipText);

		loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());

		*m_unif_planeClip = glGetUniformLocation(this->program_handler(),"planeClip");
		setClippingPlane(Geom::Vec4f (0.0f, 0.0f, 0.0f, 0.0f));
	}
	else
	{
		m_nameVS = "ShaderSimpleColor_vs";
		m_nameFS = "ShaderSimpleColor_fs";
		m_nameGS = "";

		// chose GL defines (2 or 3)
		// and compile shaders
		std::string glxvert(GLSLShader::defines_gl());
		glxvert.append(vertexShaderText);

		std::string glxfrag(GLSLShader::defines_gl());
		if (black_is_transparent)
			glxfrag.append("#define BLACK_TRANSPARENCY 1\n");
		glxfrag.append(fragmentShaderText);

		loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());
	}

	*m_unif_color = glGetUniformLocation(this->program_handler(),"color");
	//Default values
	Geom::Vec4f color(0.1f, 0.9f, 0.1f, 0.0f);
	setColor(color);
}

void ShaderSimpleColor::setColor(const Geom::Vec4f& color)
{
	m_color = color;
	bind();
	glUniform4fv(*m_unif_color, 1, color.data());
	unbind();
}


void ShaderSimpleColor::setClippingPlane(const Geom::Vec4f& plane)
{
	m_planeClip = plane;
	bind();
	glUniform4fv(*m_unif_planeClip, 1, plane.data());
	unbind();
}


unsigned int ShaderSimpleColor::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

void ShaderSimpleColor::restoreUniformsAttribs()
{
	*m_unif_color = glGetUniformLocation(this->program_handler(), "color");
	bind();
	glUniform4fv(*m_unif_color, 1, m_color.data());
	glUniform4fv(*m_unif_planeClip, 1, m_planeClip.data());
	bindVA_VBO("VertexPosition", m_vboPos);
	unbind();
}

} // namespace Utils

} // namespace CGoGN
