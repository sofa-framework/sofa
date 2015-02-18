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

#include "Utils/Shaders/shaderColorPerVertex.h"


namespace CGoGN
{

namespace Utils
{

#include "shaderColorPerVertex.vert"
#include "shaderColorPerVertex.frag"

//std::string ShaderColorPerVertex::vertexShaderText =
//		"ATTRIBUTE vec3 VertexPosition;\n"
//		"ATTRIBUTE vec3 VertexColor;\n"
//		"uniform mat4 ModelViewProjectionMatrix;\n"
//		"VARYING_VERT vec3 color;\n"
//		"INVARIANT_POS;\n"
//		"void main ()\n"
//		"{\n"
//		"	gl_Position = ModelViewProjectionMatrix * vec4 (VertexPosition, 1.0);\n"
//		"	color = VertexColor;\n"
//		"}";
//
//
//std::string ShaderColorPerVertex::fragmentShaderText =
//		"PRECISON;\n"
//		"VARYING_FRAG vec3 color;\n"
//		"FRAG_OUT_DEF;\n"
//		"void main()\n"
//		"{\n"
//		"	FRAG_OUT=vec4(color,0.0);\n"
//		"}";


ShaderColorPerVertex::ShaderColorPerVertex(bool black_is_transparent)
{
	m_nameVS = "ShaderColorPerVertex_vs";
	m_nameFS = "ShaderColorPerVertex_fs";
	m_nameGS = "ShaderColorPerVertex_gs";

	std::string glxvert(*GLSLShader::DEFINES_GL);
	glxvert.append(vertexShaderText);

	std::string glxfrag(*GLSLShader::DEFINES_GL);
	if (black_is_transparent)
		glxfrag.append("#define BLACK_TRANSPARENCY 1\n");
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());

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
	unbind();
}

void ShaderColorPerVertex::setOpacity(float op)
{
	m_opacity = op;
	bind();
	glUniform1f (*m_unif_alpha, m_opacity);
	unbind();
}

} // namespace Utils

} // namespace CGoGN
