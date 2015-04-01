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
#include "Utils/fbo.h"
#include "Utils/textureSticker.h"

namespace CGoGN
{

namespace Utils
{

// Initialize static variables and constants
bool FBO::sm_anyFboBound = false;

FBO::FBO(unsigned int width, unsigned int height)
	: m_width                 (width)
	, m_height                (height)
	, m_maxColorAttachments   (0)
	, m_fboId                 (CGoGNGLuint(0))
	, m_renderBufferId        (CGoGNGLuint(0))
	, m_colorAttachmentPoints (CGoGNGLenumTable(NULL))
	, m_bound                 (false)
{
	// Generate Fbo
	glGenFramebuffers(1, &(*m_fboId));
	
	// Get the maximum number of available color attachments
	glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &m_maxColorAttachments);
	*m_colorAttachmentPoints = new GLenum[m_maxColorAttachments];
}

FBO::~FBO()
{
	GLuint textureId;

	for (unsigned int i = 0; i < m_colorTexId.size(); i++)
	{
		textureId = *(m_colorTexId.at(i));
		if (glIsTexture(textureId))
			glDeleteTextures(1, &textureId);
	}
	
	for (unsigned int i = 0; i < m_depthTexId.size(); i++)
	{
		textureId = *(m_depthTexId.at(i));
		if (glIsTexture(textureId))
			glDeleteTextures(1, &textureId);
	}
	
	if (glIsRenderbuffer(*m_renderBufferId))
		glDeleteRenderbuffers(1, &(*m_renderBufferId));
		
	if (glIsFramebuffer(*m_fboId))
		glDeleteFramebuffers(1, &(*m_fboId));
		
	delete[] *m_colorAttachmentPoints;
}

/*
void FBO::attachRenderbuffer(GLenum internalFormat)
{
	if (sm_anyFboBound)
	{
		CGoGNerr << "FBO::AttachRenderbuffer : No Fbo should be bound when attaching a render buffer." << CGoGNendl;
		return;
	}

	GLenum attachment;
	GLuint renderBufferId;

	switch (internalFormat)
	{
		case GL_DEPTH_COMPONENT :
			attachment = GL_DEPTH_ATTACHMENT;
			break;
			
		case GL_STENCIL_INDEX :
			attachment = GL_STENCIL_ATTACHMENT;
			break;
			
		case GL_DEPTH_STENCIL :
			attachment = GL_DEPTH_STENCIL_ATTACHMENT;
			break;
			
		default :
			CGoGNerr << "FBO::AttachRenderbuffer : Bad internal format." << CGoGNendl;
			return;
			break;
	}
	
	// Delete old render buffer if it exists
	if (glIsRenderbuffer(*m_renderBufferId))
		glDeleteRenderbuffers(1, &(*m_renderBufferId));

	// Generate render buffer
	glGenRenderbuffers(1, &renderBufferId);
	glBindRenderbuffer(GL_RENDERBUFFER, renderBufferId);
	glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, m_width, m_height);
	
	// Attach render buffer to Fbo
	glBindFramebuffer(GL_FRAMEBUFFER, *m_fboId);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachment, GL_RENDERBUFFER, renderBufferId);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	*m_renderBufferId = renderBufferId;
}
*/

int FBO::createAttachColorTexture(GLenum internalFormat, GLint filter)
{
	if (sm_anyFboBound)
	{
		CGoGNerr << "FBO::AttachColorTexture : No Fbo should be bound when attaching a texture." << CGoGNendl;
		return -1;
	}
	
	if ((int) m_colorTexId.size() == m_maxColorAttachments)
	{
		CGoGNerr << "FBO::AttachColorTexture : The maximum number of color textures has been exceeded." << CGoGNendl;
		return -1;
	}

	GLenum attachment;
	GLenum format;
	GLenum type;
	GLuint textureId;

    attachment = GL_COLOR_ATTACHMENT0 + m_colorTexId.size();

	switch (internalFormat)
	{
	// four components
		case GL_RGBA :
			format = GL_RGBA;
			type = GL_FLOAT;
			break;
			
		case GL_RGBA32F :
			format = GL_RGBA;
			type = GL_FLOAT;
			break;

	// three components
		case GL_RGB :
			format = GL_RGB;
			type = GL_FLOAT;
			break;
			
		case GL_RGB32F :
			format = GL_RGB;
			type = GL_FLOAT;
			break;
			
	// two components
		case GL_RG32F:
			format = GL_RG;
			type = GL_FLOAT;
			break;

		case GL_RG:
			format = GL_RG;
			type = GL_FLOAT;
			break;
			
	// one component
		case GL_R32F:
			format = GL_RED;
			type = GL_FLOAT;
			break;

		case GL_RED:
			format = GL_RED;
			type = GL_FLOAT;
			break;
		default :
			CGoGNerr << "FBO::AttachColorTexture : Specified internal format not handled." << CGoGNendl;
			return false;
			break;
	}

	// Generate texture
	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, m_width, m_height, 0, format, type, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// Attach texture to Fbo
	glBindFramebuffer(GL_FRAMEBUFFER, *m_fboId);
	glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, textureId, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_colorTexId.push_back(CGoGNGLuint(textureId));
    unsigned int num  = m_colorTexId.size() - 1;
	(*m_colorAttachmentPoints)[num] = attachment;

	return int(num);
}

int FBO::attachColorTexture(CGoGNGLuint textureId)
{
	if (sm_anyFboBound)
	{
		CGoGNerr << "FBO::AttachColorTexture : No Fbo should be bound when attaching a texture." << CGoGNendl;
		return -1;
	}

	if ((int) m_colorTexId.size() == m_maxColorAttachments)
	{
		CGoGNerr << "FBO::AttachColorTexture : The maximum number of color textures has been exceeded." << CGoGNendl;
		return -1;
	}

	GLenum attachment;

    attachment = GL_COLOR_ATTACHMENT0 + m_colorTexId.size();

	// Attach texture to Fbo
	glBindFramebuffer(GL_FRAMEBUFFER, *m_fboId);
	glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, *textureId, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_colorTexId.push_back(textureId);
    unsigned int num  = m_colorTexId.size() - 1;
	(*m_colorAttachmentPoints)[num] = attachment;

	return int(num);
}


bool FBO::createAttachDepthTexture(GLint filter)
{
	if (sm_anyFboBound)
	{
		CGoGNerr << "FBO::AttachDepthTexture : No Fbo should be bound when attaching a texture." << CGoGNendl;
		return false;
	}
	
	if( int(m_depthTexId.size()) == 1 )
	{
		std::cout << "FBO::AttachDepthTexture : Only one depth texture can be attached." << std::endl;
		return false;
	}

	GLenum attachment;
	GLenum internalFormat;
	GLenum format;
	GLenum type;
	GLuint textureId;

	attachment = GL_DEPTH_ATTACHMENT;
	internalFormat = GL_DEPTH_COMPONENT24;
	format = GL_DEPTH_COMPONENT;
	type = GL_FLOAT;

	// Generate texture
	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, m_width, m_height, 0, format, type, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// Attach texture to Fbo
	glBindFramebuffer(GL_FRAMEBUFFER, *m_fboId);
	glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, textureId, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_depthTexId.push_back(CGoGNGLuint(textureId));
	return true;
}


bool FBO::attachDepthTexture(CGoGNGLuint textureId)
{
	if (sm_anyFboBound)
	{
		CGoGNerr << "FBO::AttachDepthTexture : No Fbo should be bound when attaching a texture." << CGoGNendl;
		return false;
	}

	if( int(m_depthTexId.size()) == 1 )
	{
		std::cout << "FBO::AttachDepthTexture : Only one depth texture can be attached." << std::endl;
		return false;
	}

	GLenum attachment = GL_DEPTH_ATTACHMENT;

	// Attach texture to Fbo
	glBindFramebuffer(GL_FRAMEBUFFER, *m_fboId);
	glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, *textureId, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_depthTexId.push_back(textureId);
	return true;
}


void FBO::enableAllColorAttachments()
{
	if (!m_bound)
	{
		CGoGNerr << "FBO::EnableColorAttachments : Fbo must be bound when enabling color attachments." << CGoGNendl;
		return;
	}

	glDrawBuffers(GLsizei(m_colorTexId.size()), *m_colorAttachmentPoints);
}

void FBO::enableColorAttachment(int num)
{
	if (!m_bound)
	{
		CGoGNerr << "FBO::EnableColorAttachments : Fbo must be bound when enabling color attachments." << CGoGNendl;
		return;
	}

	glDrawBuffers(1, &(*m_colorAttachmentPoints)[num]);
}

CGoGNGLuint FBO::getColorTexId(int num)
{
	if ((int) m_colorTexId.size() > num)
		return m_colorTexId[num];
	else
		return CGoGNGLuint(0);
}

CGoGNGLuint FBO::getDepthTexId()
{
	if ((int) m_depthTexId.size() > 0)
		return m_depthTexId[0];
	else
		return CGoGNGLuint(0);
}

void FBO::draw(int attach)
{
	Utils::TextureSticker::fullScreenTexture(this->getColorTexId(attach));
}

void FBO::drawWithDepth(int attach)
{
	Utils::TextureSticker::fullScreenTextureDepth(this->getColorTexId(attach),this->getDepthTexId());
}

void FBO::drawWithDepth(int attach, CGoGNGLuint depthTexId)
{
	Utils::TextureSticker::fullScreenTextureDepth(this->getColorTexId(attach), depthTexId);
}


void FBO::draw(Utils::GLSLShader* shader)
{
	Utils::TextureSticker::fullScreenShader(shader);
}

void FBO::bind()
{
	if (m_bound)
	{
		CGoGNerr << "FBO::Bind : This Fbo is already bound." << CGoGNendl;
		return;
	}

	if (sm_anyFboBound)
	{
		CGoGNerr << "FBO::Bind : Only one Fbo can be bound at the same time." << CGoGNendl;
		return;
	}

	// Bind this Fbo
	glBindFramebuffer(GL_FRAMEBUFFER, *m_fboId);
	m_bound = true;
	sm_anyFboBound = true;
	
	// Get current viewport
	glGetIntegerv(GL_VIEWPORT, m_oldViewport);
	
	// Set the viewport to the size of the Fbo
	glViewport(0, 0, m_width, m_height);
}

void FBO::unbind()
{
	if (m_bound)
	{
		// Unbind this Fbo
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		m_bound = false;
		sm_anyFboBound = false;
		
		// Reset the viewport to the main framebuffer size
		glViewport(m_oldViewport[0], m_oldViewport[1], m_oldViewport[2], m_oldViewport[3]);
	}
}

void FBO::safeUnbind()
{
	glFlush();
	unbind();
	glDrawBuffer(GL_BACK);
}

void FBO::checkFBO()
{
	if (sm_anyFboBound)
	{
		CGoGNerr << "FBO::CheckFBO : No Fbo should be bound when checking a Fbo's status." << CGoGNendl;
		return;
	}

	GLenum status;

	// Get Fbo status
	glBindFramebuffer(GL_FRAMEBUFFER, *m_fboId);
	status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	if (status != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Fbo status error : " << status << std::endl;

    switch (status) {
    case GL_FRAMEBUFFER_COMPLETE:
    	break;
    case GL_FRAMEBUFFER_UNDEFINED:
    	std::cout << "GL_FRAMEBUFFER_UNDEFINED" << std::endl;
    	break;
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
    	std::cout << "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT" << std::endl;
    	break;
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
    	std::cout << "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT" << std::endl;
    	break;
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
    	std::cout << "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER" << std::endl;
    	break;
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
    	std::cout << "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER" << std::endl;
    	break;
    case GL_FRAMEBUFFER_UNSUPPORTED:
    	std::cout << "GL_FRAMEBUFFER_UNSUPPORTED" << std::endl;
    	break;
    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
    	std::cout << "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE" << std::endl;
    	break;
    case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:
    	std::cout << "GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS" << std::endl;
    	break;
    }
}

unsigned int FBO::getWidth() const
{
	return m_width;
}

unsigned int FBO::getHeight() const
{
	return m_height;
}

} // namespace Utils

} // namespace CGoGN

