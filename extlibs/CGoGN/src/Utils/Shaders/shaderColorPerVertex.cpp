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
#include "Utils/Shaders/shaderColorPerVertex.h"


namespace CGoGN
{

namespace Utils
{

#include "shaderColorPerVertex.vert"
#include "shaderColorPerVertex.frag"
#include "shaderColorPerVertexClip.vert"
#include "shaderColorPerVertexClip.frag"


ShaderColorPerVertex::ShaderColorPerVertex(bool withClipping, bool black_is_transparent)
{
	if (withClipping)
	{
		m_nameVS = "ShaderColorPerVertexClip_vs";
		m_nameFS = "ShaderColorPerVertexClip_fs";
		m_nameGS = "";

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
		m_nameVS = "ShaderColorPerVertex_vs";
		m_nameFS = "ShaderColorPerVertex_fs";
		m_nameGS = "ShaderColorPerVertex_gs";

		std::string glxvert(GLSLShader::defines_gl());
		glxvert.append(vertexShaderText);

		std::string glxfrag(GLSLShader::defines_gl());
		if (black_is_transparent)
			glxfrag.append("#define BLACK_TRANSPARENCY 1\n");
		glxfrag.append(fragmentShaderText);

		loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());

	}
	bind();
	*m_unif_alpha = glGetUniformLocation(this->program_handler(), "alpha");
	glUniform1f (*m_unif_alpha, 1.0f);
	m_opacity = 1.0f;
	unbind();

}

unsigned int ShaderColorPerVertex::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int ShaderColorPerVertex::setAttributeColor(VBO* vbo)
{
	m_vboCol = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexColor", vbo);
	unbind();
	return id;
}


void ShaderColorPerVertex::restoreUniformsAttribs()
{
	bind();
	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexColor", m_vboCol);
	glUniform1f (*m_unif_alpha, m_opacity);
	glUniform4fv(*m_unif_planeClip, 1, m_planeClip.data());
	unbind();
}

void ShaderColorPerVertex::setOpacity(float op)
{
	m_opacity = op;
	bind();
	glUniform1f (*m_unif_alpha, m_opacity);
	unbind();
}

void ShaderColorPerVertex::setClippingPlane(const Geom::Vec4f& plane)
{
	m_planeClip = plane;
	bind();
	glUniform4fv(*m_unif_planeClip, 1, plane.data());
	unbind();
}



} // namespace Utils

} // namespace CGoGN
