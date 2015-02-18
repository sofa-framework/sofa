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

#include "GL/glew.h"
#include "Utils/textureSticker.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace CGoGN
{

namespace Utils
{

// Initialize static variables and constants
bool TextureSticker::sm_isInitialized = false;
Utils::VBO* TextureSticker::sm_quadPositionsVbo = NULL;
Utils::VBO* TextureSticker::sm_quadTexCoordsVbo = NULL;
Utils::ShaderSimpleTexture* TextureSticker::sm_textureMappingShader = NULL;
Utils::ShaderTextureDepth* TextureSticker::sm_depthtextureMappingShader = NULL;

TextureSticker::TextureSticker()
{
}

TextureSticker::~TextureSticker()
{
	if (sm_textureMappingShader!= NULL)
		delete sm_textureMappingShader;

	if (sm_depthtextureMappingShader!= NULL)
		delete sm_depthtextureMappingShader;
}

void TextureSticker::fullScreenTexture(CGoGNGLuint texId)
{
	glClear(GL_COLOR_BUFFER_BIT);

	// Check if TextureSticker's elements have been initialized before
	if (!sm_isInitialized)
	{
		initializeElements();
		sm_isInitialized = true;
	}
	
	// Check if depth test is enabled
	GLboolean wasDepthTestEnabled = glIsEnabled(GL_DEPTH_TEST);
	
	// Disable depth test if it was enabled
	if (wasDepthTestEnabled == GL_TRUE)
		glDisable(GL_DEPTH_TEST);

	// Bind texture mapping shader
	sm_textureMappingShader->bind();
	
	// Set texture uniform
	sm_textureMappingShader->setTextureUnit(GL_TEXTURE0);
	sm_textureMappingShader->activeTexture(texId);
	
	// Set matrices uniforms
	glm::mat4 projMatrix = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
	glm::mat4 viewMatrix(1.0f);
	sm_textureMappingShader->updateMatrices(projMatrix, viewMatrix);

	// Draw quad
	sm_textureMappingShader->enableVertexAttribs();
	glDrawArrays(GL_QUADS, 0, 4);
	sm_textureMappingShader->disableVertexAttribs();

	// Unbind texture mapping shader
	sm_textureMappingShader->unbind();

	// Re-enable depth test if it was enabled before
	if (wasDepthTestEnabled == GL_TRUE)
		glEnable(GL_DEPTH_TEST);
}


void TextureSticker::fullScreenTextureDepth(CGoGNGLuint texId, CGoGNGLuint dtexId)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Check if TextureSticker's elements have been initialized before
	if (!sm_isInitialized)
	{
		initializeElements();
		sm_isInitialized = true;
	}

	// Check if depth test is enabled

//	GLboolean wasDepthTestEnabled = glIsEnabled(GL_DEPTH_TEST);

	// Disable depth test if it was enabled
//	if (wasDepthTestEnabled == GL_TRUE)
//		glDisable(GL_DEPTH_TEST);

	// Bind texture mapping shader
	sm_depthtextureMappingShader->bind();

	// Set texture uniform
	sm_depthtextureMappingShader->setTextureUnit(GL_TEXTURE0);
	sm_depthtextureMappingShader->activeTexture(texId);
	sm_depthtextureMappingShader->setDepthTextureUnit(GL_TEXTURE1);
	sm_depthtextureMappingShader->activeDepthTexture(dtexId);


	// Set matrices uniforms
	glm::mat4 projMatrix = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
	glm::mat4 viewMatrix(1.0f);
	sm_depthtextureMappingShader->updateMatrices(projMatrix, viewMatrix);

	// Draw quad
	sm_depthtextureMappingShader->enableVertexAttribs();
	glDrawArrays(GL_QUADS, 0, 4);
	sm_depthtextureMappingShader->disableVertexAttribs();

	// Unbind texture mapping shader
	sm_depthtextureMappingShader->unbind();

	// Re-enable depth test if it was enabled before
//	if (wasDepthTestEnabled == GL_TRUE)
//		glEnable(GL_DEPTH_TEST);
}





void TextureSticker::fullScreenShader(Utils::GLSLShader* shader)
{
	// Check if TextureSticker's elements have been initialized before
	if (!sm_isInitialized)
	{
		initializeElements();
		sm_isInitialized = true;
	}
	
	// Check if depth test is enabled
	GLboolean wasDepthTestEnabled = glIsEnabled(GL_DEPTH_TEST);
	
	// Disable depth test if it was enabled
	if (wasDepthTestEnabled == GL_TRUE)
		glDisable(GL_DEPTH_TEST);
	
	// Bind shader
	shader->bind();
	
	// Set matrices uniforms
	glm::mat4 projMatrix = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
	glm::mat4 viewMatrix(1.0f);
	shader->updateMatrices(projMatrix, viewMatrix);

	// Draw quad
	shader->enableVertexAttribs();
	glDrawArrays(GL_QUADS, 0, 4);
	shader->disableVertexAttribs();
	
	// Unbind shader
	shader->unbind();

	// Re-enable depth test if it was enabled before
	if (wasDepthTestEnabled == GL_TRUE)
		glEnable(GL_DEPTH_TEST);
}

VBO* TextureSticker::getPositionsVbo()
{
	// Check if TextureSticker's elements have been initialized before
	if (!sm_isInitialized)
	{
		initializeElements();
		sm_isInitialized = true;
//		return NULL;
	}
	
	return sm_quadPositionsVbo;
}

VBO* TextureSticker::getTexCoordsVbo()
{
	// Check if TextureSticker's elements have been initialized before
	if (!sm_isInitialized)
	{
		initializeElements();
		sm_isInitialized = true;
//		return NULL;
	}
	
	return sm_quadTexCoordsVbo;
}

void TextureSticker::initializeElements()
{
	// Initialize positions and texture coords Vbos
	
	sm_quadPositionsVbo = new Utils::VBO();
	sm_quadTexCoordsVbo = new Utils::VBO();
	sm_quadPositionsVbo->setDataSize(3);
	sm_quadTexCoordsVbo->setDataSize(2);
	sm_quadPositionsVbo->allocate(4);
	sm_quadTexCoordsVbo->allocate(4);

	GLfloat positions[] = {
		-1.0f, -1.0f, 0.0f,
		+1.0f, -1.0f, 0.0f,
		+1.0f, +1.0f, 0.0f,
		-1.0f, +1.0f, 0.0f
		};
	GLfloat texCoords[] = {
		0.0f, 0.0f,
		1.0f, 0.0f,
		1.0f, 1.0f,
		0.0f, 1.0f
		};
		
	GLfloat* positionsPtr = (GLfloat*) sm_quadPositionsVbo->lockPtr();
	memcpy(positionsPtr, positions, 3 * 4 * sizeof(GLfloat));
	sm_quadPositionsVbo->releasePtr();
	GLfloat* texCoordsPtr = (GLfloat*) sm_quadTexCoordsVbo->lockPtr();
	memcpy(texCoordsPtr, texCoords, 2 * 4 * sizeof(GLfloat));
	sm_quadTexCoordsVbo->releasePtr();
	
	// Initialize simple texture mapping shader
	sm_textureMappingShader = new Utils::ShaderSimpleTexture();
	sm_textureMappingShader->setAttributePosition(sm_quadPositionsVbo);
	sm_textureMappingShader->setAttributeTexCoord(sm_quadTexCoordsVbo);

	sm_depthtextureMappingShader = new Utils::ShaderTextureDepth();
	sm_depthtextureMappingShader->setAttributePosition(sm_quadPositionsVbo);
	sm_depthtextureMappingShader->setAttributeTexCoord(sm_quadTexCoordsVbo);

}

} // namespace Utils

} // namespace CGoGN

