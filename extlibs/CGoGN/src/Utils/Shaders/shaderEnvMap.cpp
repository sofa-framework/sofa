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
#include "Utils/Shaders/shaderEnvMap.h"
#include <string.h>

namespace CGoGN
{

namespace Utils
{

#include "shaderEnvMap.vert"
#include "shaderEnvMap.frag"


ShaderEnvMap::ShaderEnvMap(bool doubleSided, bool withEyePosition):
	m_with_color(false),
	m_with_eyepos(withEyePosition),
	m_ambiant(Geom::Vec4f(0.05f,0.05f,0.1f,0.0f)),
	m_diffuse(Geom::Vec4f(0.1f,1.0f,0.1f,0.0f)),
	m_blend(0.5f),
	m_lightPos(Geom::Vec3f(10.0f,10.0f,10000.0f)),
	m_vboPos(NULL),
	m_vboNormal(NULL),
	m_vboColor(NULL)
{

	m_nameVS = "ShaderEnvMap_vs";
	m_nameFS = "ShaderEnvMap_fs";

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

	std::cout << "init texture" << std::endl;
	glEnable(GL_TEXTURE_CUBE_MAP);
	glGenTextures(1, &(*m_texId));
	glBindTexture(GL_TEXTURE_CUBE_MAP, *m_texId);

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

}

void ShaderEnvMap::getLocations()
{
	bind();
	*m_unif_ambiant   = glGetUniformLocation(this->program_handler(), "materialAmbient");
	*m_unif_diffuse   = glGetUniformLocation(this->program_handler(), "materialDiffuse");
	*m_unif_blend = glGetUniformLocation(this->program_handler(), "blendCoef");
	*m_unif_lightPos  = glGetUniformLocation(this->program_handler(), "lightPosition");
	if (m_with_eyepos)
		*m_unif_eyePos= glGetUniformLocation(this->program_handler(), "eyePosition");

	*m_unif_envMap	  = glGetUniformLocation(this->program_handler(), "EnvMap");
	unbind();
}

void ShaderEnvMap::sendParams()
{
	bind();
	glUniform4fv(*m_unif_ambiant,  1, m_ambiant.data());
	glUniform4fv(*m_unif_diffuse,  1, m_diffuse.data());
	glUniform1f(*m_unif_blend,    m_blend);
	glUniform3fv(*m_unif_lightPos, 1, m_lightPos.data());
	if (m_with_eyepos)
		glUniform3fv(*m_unif_eyePos, 1, m_eyePos.data());
	// we use texture engine 0
	glUniform1i(*m_unif_envMap,GL_TEXTURE0);
	unbind();
}

void ShaderEnvMap::setAmbiant(const Geom::Vec4f& ambiant)
{
	m_ambiant = ambiant;
	bind();
	glUniform4fv(*m_unif_ambiant,1, m_ambiant.data());
	unbind();
}

void ShaderEnvMap::setDiffuse(const Geom::Vec4f& diffuse)
{
	m_diffuse = diffuse;
	bind();
	glUniform4fv(*m_unif_diffuse,1, m_diffuse.data());
	unbind();
}


void ShaderEnvMap::setBlendCoef(float blend)
{
	m_blend = blend;
	bind();
	glUniform1f (*m_unif_blend, m_blend);
	unbind();
}

void ShaderEnvMap::setLightPosition(const Geom::Vec3f& lightPos)
{
	m_lightPos = lightPos;
	bind();
	glUniform3fv(*m_unif_lightPos, 1, m_lightPos.data());
	unbind();
}

void ShaderEnvMap::setEyePosition(const Geom::Vec3f& eyePos)
{
	if (m_with_eyepos)
	{
		m_eyePos = eyePos;
		bind();
		glUniform3fv(*m_unif_eyePos, 1, m_eyePos.data());
		unbind();
	}
}

void ShaderEnvMap::setParams(const Geom::Vec4f& ambiant, const Geom::Vec4f& diffuse, float blend, const Geom::Vec3f& lightPos)
{
	m_ambiant = ambiant;
	m_diffuse = diffuse;
	m_blend = blend;
	m_lightPos = lightPos;
	sendParams();
}

unsigned int ShaderEnvMap::setAttributeColor(VBO* vbo)
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

void ShaderEnvMap::unsetAttributeColor()
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

void ShaderEnvMap::restoreUniformsAttribs()
{
	getLocations();
	sendParams();

	bind();
	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexNormal", m_vboNormal);
	if (m_vboColor)
		bindVA_VBO("VertexColor", m_vboColor);
	unbind();
}

unsigned int ShaderEnvMap::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int ShaderEnvMap::setAttributeNormal(VBO* vbo)
{
	m_vboNormal = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexNormal", vbo);
	unbind();
	return id;
}


void ShaderEnvMap::setCubeMap(unsigned int sz,
							unsigned char* Xpos, unsigned char* Ypos, unsigned char* Zpos,
							unsigned char* Xneg, unsigned char* Yneg, unsigned char* Zneg)
{
	glEnable(GL_TEXTURE_CUBE_MAP);
	glBindTexture(GL_TEXTURE_CUBE_MAP, *m_texId);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, Xpos);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, Xneg);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, Ypos);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, Yneg);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, Zpos);
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, Zneg);
}


bool ShaderEnvMap::setCubeMapImg(unsigned int sz, unsigned char *img)
{
	glEnable(GL_TEXTURE_CUBE_MAP);
	glBindTexture(GL_TEXTURE_CUBE_MAP, *m_texId);

	unsigned char* texture = new unsigned char[3*sz*sz];

	// Y+
	unsigned char* ptrD = texture;
	unsigned char* ptrS = img + 3*sz;
	for (unsigned int i=0; i<sz; ++i)
	{
		 memcpy(ptrD, ptrS, 3*sz);
		 ptrD += 3*sz;
		 ptrS += 12*sz;
	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

//
//	// Y-
	ptrD = texture;
	ptrS = img + 24*sz*sz + 3*sz ;
	for (unsigned int i=0; i<sz; ++i)
	{
		 memcpy(ptrD, ptrS, 3*sz);
		 ptrD += 3*sz;
		 ptrS += 12*sz;
	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	// X-
	ptrD = texture;
	ptrS = img + 12*sz*sz;
	for (unsigned int i=0; i<sz; ++i)
	{
		 memcpy(ptrD, ptrS, 3*sz);
		 ptrD += 3*sz;
		 ptrS += 12*sz;
	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	// X+
	ptrD = texture;
	ptrS = img + 12*sz*sz+ 6*sz ;
	for (unsigned int i=0; i<sz; ++i)
	{
		 memcpy(ptrD, ptrS, 3*sz);
		 ptrD += 3*sz;
		 ptrS += 12*sz;
	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	// Z-
	ptrD = texture;
	ptrS = img + 12*sz*sz+ 3*sz ;
	for (unsigned int i=0; i<sz; ++i)
	{
		 memcpy(ptrD, ptrS, 3*sz);
		 ptrD += 3*sz;
		 ptrS += 12*sz;
	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	// Z+
	ptrD = texture;
	ptrS = img + 12*sz*sz+ 9*sz ;
	for (unsigned int i=0; i<sz; ++i)
	{
		 memcpy(ptrD, ptrS, 3*sz);
		 ptrD += 3*sz;
		 ptrS += 12*sz;
	}
////	for (unsigned int i = 0; i< sz*sz; ++i)
////	{
////		if (texture[3*i+2] != 0)
////			std::cout << "PIX= "<< int(texture[3*i]) << " / "<< int(texture[3*i+1]) << " / "<< int(texture[3*i+2]) << std::endl;
////	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);


	delete[] texture;
	return true;
}


bool ShaderEnvMap::setCubeMapColored()
{
	glEnable(GL_TEXTURE_CUBE_MAP);
	glBindTexture(GL_TEXTURE_CUBE_MAP, *m_texId);

	unsigned int sz = 256;
	unsigned char* texture = new unsigned char[3*sz*sz];

	for (unsigned int i = 0; i< sz*sz; ++i)
	{	
		unsigned int j = (i/sz)/8;
		unsigned int k = (i%sz)/8;
		unsigned char val=255;
		if ((j+k)%2 ==0)
			val /= 2;
		texture[i*3]   = val;
		texture[i*3+1] = 0;
		texture[i*3+2] = 0;
	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	for (unsigned int i = 0; i< sz*sz; ++i)
	{
		unsigned int j = (i/sz)/8;
		unsigned int k = (i%sz)/8;
		unsigned char val=128;
		if ((j+k)%2 ==0)
			val /= 2;
		texture[i*3]   = val;
		texture[i*3+1] = 0;
		texture[i*3+2] = 0;
	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);


	for (unsigned int i = 0; i< sz*sz; ++i)
	{
		unsigned int j = (i/sz)/8;
		unsigned int k = (i%sz)/8;
		unsigned char val=255;
		if ((j+k)%2 ==0)
			val /= 2;
		texture[i*3]   = 0;
		texture[i*3+1] = val;
		texture[i*3+2] = 0;
	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	for (unsigned int i = 0; i< sz*sz; ++i)
	{
		unsigned int j = (i/sz)/8;
		unsigned int k = (i%sz)/8;
		unsigned char val=128;
		if ((j+k)%2 ==0)
			val /= 2;
		texture[i*3]   = 0;
		texture[i*3+1] = val;
		texture[i*3+2] = 0;
	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);


	for (unsigned int i = 0; i< sz*sz; ++i)
	{
		unsigned int j = (i/sz)/8;
		unsigned int k = (i%sz)/8;
		unsigned char val=255;
		if ((j+k)%2 ==0)
			val /= 2;
		texture[i*3]   = 0;
		texture[i*3+1] = 0;
		texture[i*3+2] = val;
	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	for (unsigned int i = 0; i< sz*sz; ++i)
	{
		unsigned int j = (i/sz)/8;
		unsigned int k = (i%sz)/8;
		unsigned char val=128;
		if ((j+k)%2 ==0)
			val /= 2;
		texture[i*3]   = 0;
		texture[i*3+1] = 0;
		texture[i*3+2] = val;
	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	delete[] texture;
	return true;
}



bool ShaderEnvMap::setCubeMapCheckered()
{
	glEnable(GL_TEXTURE_CUBE_MAP);
	glBindTexture(GL_TEXTURE_CUBE_MAP, *m_texId);

	unsigned int sz = 256;
	unsigned char* texture = new unsigned char[3*sz*sz];

	unsigned int width = 16;

	for (unsigned int i = 0; i< sz; ++i)
	{
		for (unsigned int j = 0; j< sz; ++j)
		{

			unsigned int k = i*sz+j;
			if ((i/width)%2 == (j/width)%2)
			{
				texture[k*3]   = 255;
				texture[k*3+1] = 255;
				texture[k*3+2] = 255;
			}
			else
			{
				texture[k*3]   = 0;
				texture[k*3+1] = 0;
				texture[k*3+2] = 0;
			}
		}
	}
	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	glTexImage2D(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, GL_RGBA, sz, sz, 0, GL_RGB, GL_UNSIGNED_BYTE, texture);

	delete[] texture;
	return true;
}


#ifdef CGOGN_WITH_QT
bool ShaderEnvMap::setCubeMap(const std::string& filename)
{

	QImage* ptr = new QImage(filename.c_str());

	if (ptr == NULL)
	{
		CGoGNout << "Impossible to load "<< filename << CGoGNendl;
		return false;
	}


	if ( ptr->width()/4 != ptr->height()/3 )
	{
		CGoGNout << "wrong image configuration for envMap  W/4 != H/3 " << CGoGNendl;
		delete ptr;
		return false;
	}

	QImage img = ptr->convertToFormat(QImage::Format_RGB888);

	std::cout << "IMAGE: "<< img.width()/4 << std::endl;

	unsigned char *pix = img.bits();
	setCubeMapImg(img.width()/4,pix);

	delete ptr;
	return true;
}
#endif



void ShaderEnvMap::predraw()
{
	glActiveTexture(GL_TEXTURE0);
	glEnable(GL_TEXTURE_CUBE_MAP);
	glBindTexture(GL_TEXTURE_2D, *m_texId);
}

void ShaderEnvMap::postdraw()
{
	glDisable(GL_TEXTURE_CUBE_MAP);
}

} // namespace Utils

} // namespace CGoGN
