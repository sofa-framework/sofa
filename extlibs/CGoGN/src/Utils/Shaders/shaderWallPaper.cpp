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
#include "Utils/Shaders/shaderWallPaper.h"

namespace CGoGN
{

namespace Utils
{

#include "shaderWallPaper.vert"
#include "shaderWallPaper.frag"

ShaderWallPaper::ShaderWallPaper():
	m_tex_ptr(NULL)
{
	std::string glxvert(GLSLShader::defines_gl());
	glxvert.append(vertexShaderText);

	std::string glxfrag(GLSLShader::defines_gl());
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());
	
	m_unif_unit = glGetUniformLocation(this->program_handler(), "textureUnit");
	m_unif_pos = glGetUniformLocation(this->program_handler(), "pos");
	m_unif_sz = glGetUniformLocation(this->program_handler(), "sz");

	m_vboPos = new Utils::VBO();
	m_vboPos->setDataSize(2);
	m_vboPos->allocate(4);
	Geom::Vec2f* ptrPos = reinterpret_cast<Geom::Vec2f*>(m_vboPos->lockPtr());

	ptrPos[0] = Geom::Vec2f(0.0f, 0.0f);
	ptrPos[1] = Geom::Vec2f(1.0f, 0.0f);
	ptrPos[2] = Geom::Vec2f(1.0f, 1.0f);
	ptrPos[3] = Geom::Vec2f(0.0f, 1.0f);

//	ptrPos[0] = Geom::Vec3f(-1,-1, 0.9999f);
//	ptrPos[1] = Geom::Vec3f( 1,-1, 0.9999f);
//	ptrPos[2] = Geom::Vec3f( 1, 1, 0.9999f);
//	ptrPos[3] = Geom::Vec3f(-1, 1, 0.9999f);

	m_vboPos->releasePtr();

	bindVA_VBO("VertexPosition", m_vboPos);

	m_vboTexCoord = new Utils::VBO();
	m_vboTexCoord->setDataSize(2);

	m_vboTexCoord = new Utils::VBO();
	m_vboTexCoord->setDataSize(2);
	m_vboTexCoord->allocate(4);
	Geom::Vec2f* ptrTex = reinterpret_cast<Geom::Vec2f*>(m_vboTexCoord->lockPtr());

	ptrTex[0] = Geom::Vec2f(0.0f, 0.0f);
	ptrTex[1] = Geom::Vec2f(1.0f, 0.0f);
	ptrTex[2] = Geom::Vec2f(1.0f, 1.0f);
	ptrTex[3] = Geom::Vec2f(0.0f, 1.0f);

	m_vboTexCoord->releasePtr();

	bindVA_VBO("VertexTexCoord", m_vboTexCoord);
}

ShaderWallPaper::~ShaderWallPaper()
{
	delete m_vboPos;
	delete m_vboTexCoord;
}

void ShaderWallPaper::setTextureUnit(GLenum texture_unit)
{
	int unit = texture_unit - GL_TEXTURE0;
	m_unit = unit;
	bind();
	glUniform1i(*m_unif_unit, unit);
	unbind();
}

void ShaderWallPaper::setTexture(Utils::GTexture* tex)
{
	m_tex_ptr = tex;
}

void ShaderWallPaper::activeTexture()
{
	glActiveTexture(GL_TEXTURE0 + m_unit);
	m_tex_ptr->bind();
}

void ShaderWallPaper::activeTexture(CGoGNGLuint texId)
{
	glActiveTexture(GL_TEXTURE0 + m_unit);
	glBindTexture(GL_TEXTURE_2D, *texId);
}

void ShaderWallPaper::restoreUniformsAttribs()
{
	m_unif_unit = glGetUniformLocation(this->program_handler(), "textureUnit");

	bind();
	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexTexCoord", m_vboTexCoord);
	glUniform1i(*m_unif_unit, m_unit);
	unbind();
}

void ShaderWallPaper::draw()
{
	float pos[2];
	pos[0] = -1.0f ;
	pos[1] = -1.0f ;
	float sz[2];
	sz[0] = 2.0f;
	sz[1] = 2.0f;

	bind();
	glUniform2fv(*m_unif_pos, 1, pos);
	glUniform2fv(*m_unif_sz, 1, sz);
	activeTexture();
	unbind();

	enableVertexAttribs();
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	disableVertexAttribs();
}

void ShaderWallPaper::drawBack(int window_w, int window_h, int x, int y, int w, int h, Utils::GTexture* button)
{
	// tranform position to -1,1 and invert Y ( in GL O in left,bottom)
	float sz[2];
	sz[0] = float(2*w)/float(window_w);
	sz[1] = float(2*h)/float(window_h);

	float pos[2];
	pos[0] = -1.0f + float(2*x)/float(window_w);
	pos[1] = 1.0f - float(2*y)/float(window_h) - sz[1];

	this->bind();
	glUniform2fv(*m_unif_pos,1, pos);
	glUniform2fv(*m_unif_sz,1, sz);
	this->unbind();

	setTexture(button);
	activeTexture();

	enableVertexAttribs();
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	disableVertexAttribs();
}

void ShaderWallPaper::drawFront(int window_w, int window_h, int x, int y, int w, int h, Utils::GTexture* button)
{
	// tranform position to -1,1 and invert Y (in GL O in left,bottom)
	float sz[2];
	sz[0] = float(2*w)/float(window_w);
	sz[1] = float(2*h)/float(window_h);

	float pos[2];
	pos[0] = -1.0f + float(2*x)/float(window_w);
	pos[1] = 1.0f - float(2*y)/float(window_h) - sz[1];

	bind();
	glUniform2fv(*m_unif_pos,1, pos);
	glUniform2fv(*m_unif_sz,1, sz);
	unbind();

	setTexture(button);
	activeTexture();

	glDisable(GL_DEPTH_TEST);

	enableVertexAttribs();
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	disableVertexAttribs();

	glEnable(GL_DEPTH_TEST);
}

} // namespace Utils

} // namespace CGoGN
