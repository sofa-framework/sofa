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
#include "Utils/Shaders/shaderPhongTexture.h"

namespace CGoGN
{

namespace Utils
{

#include "shaderPhongTexture.vert"
#include "shaderPhongTexture.frag"


ShaderPhongTexture::ShaderPhongTexture(bool doubleSided, bool withEyePosition):
	m_with_eyepos(withEyePosition),
	m_ambient(0.1f),
	m_specular(Geom::Vec4f(1.0f,1.0f,1.0f,0.0f)),
	m_shininess(100.0f),
	m_lightPos(Geom::Vec3f(10.0f,10.0f,1000.0f)),
	m_vboPos(NULL),
	m_vboNormal(NULL),
	m_vboTexCoord(NULL)
{
	m_nameVS = "ShaderPhongTexture_vs";
	m_nameFS = "ShaderPhongTexture_fs";

	// get choose GL defines (2 or 3)
	// ans compile shaders
	std::string glxvert(GLSLShader::defines_gl());
	if (m_with_eyepos)
		glxvert.append("#define WITH_EYEPOSITION");
	glxvert.append(vertexShaderText);
	std::string glxfrag(GLSLShader::defines_gl());
	// Use double sided lighting if set
	if (doubleSided)
		glxfrag.append("#define DOUBLE_SIDED\n");
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());

	// and get and fill uniforms
	getLocations();
	sendParams();
}

void ShaderPhongTexture::getLocations()
{
	bind();
	*m_unif_unit 	  = glGetUniformLocation(this->program_handler(), "textureUnit");	
	*m_unif_ambient   = glGetUniformLocation(this->program_handler(), "ambientCoef");
	*m_unif_specular  = glGetUniformLocation(this->program_handler(), "materialSpecular");
	*m_unif_shininess = glGetUniformLocation(this->program_handler(), "shininess");
	*m_unif_lightPos  = glGetUniformLocation(this->program_handler(), "lightPosition");
	if (m_with_eyepos)
		*m_unif_eyePos  = glGetUniformLocation(this->program_handler(), "eyePosition");
	unbind();
}

void ShaderPhongTexture::sendParams()
{
	bind();
	glUniform1i (*m_unif_unit,m_unit);
	glUniform1f(*m_unif_ambient,  m_ambient);
	glUniform4fv(*m_unif_specular, 1, m_specular.data());
	glUniform1f (*m_unif_shininess,    m_shininess);
	glUniform3fv(*m_unif_lightPos, 1, m_lightPos.data());
	if (m_with_eyepos)
		glUniform3fv(*m_unif_eyePos, 1, m_eyePos.data());
	unbind();
}

void ShaderPhongTexture::setTextureUnit(GLenum texture_unit)
{
	bind();
	int unit = texture_unit - GL_TEXTURE0;
	glUniform1i(*m_unif_unit, unit);
	m_unit = unit;
	unbind();
}

void ShaderPhongTexture::setTexture(Utils::GTexture* tex)
{
	m_tex_ptr = tex;
}

void ShaderPhongTexture::activeTexture()
{
	glActiveTexture(GL_TEXTURE0 + m_unit);
	m_tex_ptr->bind();
}

void ShaderPhongTexture::activeTexture(CGoGNGLuint texId)
{
	glActiveTexture(GL_TEXTURE0 + m_unit);
	glBindTexture(GL_TEXTURE_2D, *texId);
}


void ShaderPhongTexture::setAmbient(float ambient)
{
	bind();
	glUniform1f(*m_unif_ambient, ambient);
	m_ambient = ambient;
	unbind();
}

void ShaderPhongTexture::setSpecular(const Geom::Vec4f& specular)
{
	bind();
	glUniform4fv(*m_unif_specular,1,specular.data());
	m_specular = specular;
	unbind();
}

void ShaderPhongTexture::setShininess(float shininess)
{
	bind();
	glUniform1f (*m_unif_shininess, shininess);
	m_shininess = shininess;
	unbind();
}

void ShaderPhongTexture::setLightPosition(const Geom::Vec3f& lightPos)
{
	bind();
	glUniform3fv(*m_unif_lightPos,1,lightPos.data());
	m_lightPos = lightPos;
	unbind();
}

void ShaderPhongTexture::setEyePosition(const Geom::Vec3f& eyePos)
{
	if (m_with_eyepos)
	{
		bind();
		glUniform3fv(*m_unif_eyePos,1,eyePos.data());
		m_eyePos = eyePos;
		unbind();
	}
}

void ShaderPhongTexture::setParams(float ambient, const Geom::Vec4f& specular, float shininess, const Geom::Vec3f& lightPos)
{
	m_ambient = ambient;
	m_specular = specular;
	m_shininess = shininess;
	m_lightPos = lightPos;
	sendParams();
}


void ShaderPhongTexture::restoreUniformsAttribs()
{
	getLocations();
	sendParams();

	bind();
	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexNormal", m_vboNormal);
	bindVA_VBO("VertexTexCoord", m_vboTexCoord);
	unbind();
}

unsigned int ShaderPhongTexture::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int ShaderPhongTexture::setAttributeNormal(VBO* vbo)
{
	m_vboNormal = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexNormal", vbo);
	unbind();
	return id;
}

unsigned int ShaderPhongTexture::setAttributeTexCoord(VBO* vbo)
{
	m_vboTexCoord = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexTexCoord", vbo);
	unbind();
	return id;
}


} // namespace Utils

} // namespace CGoGN
