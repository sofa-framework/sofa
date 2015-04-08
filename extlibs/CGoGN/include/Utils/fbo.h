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

#ifndef __CGoGN_GLSL_FBO__
#define __CGoGN_GLSL_FBO__

#include <GL/glew.h>
#include "Utils/gl_def.h"
#include "Utils/cgognStream.h"

#include <iostream>
#include <vector>

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

// forward declaration
class GLSLShader;

/**
 * Simple Fbo class to do offscreen rendering in OpenGL.
 */
class CGoGN_UTILS_API FBO
{

public:

	/**
	 * Constructor.
	 * \param  width   Width of the Fbo in pixels
	 * \param  heigth  Height of the Fbo in pixels
	 */
	FBO(unsigned int width, unsigned int height);

	/**
	 * Destructor.
	 */
	~FBO();

	/**
	 * Attach a render buffer to the Fbo.\n
	 * If another render buffer is already attached, it will be deleted and replaced with the new one.\n
	 * When calling this function, no Fbo should be bound.\n
	 * \param  internalFormat  Internal format of the render buffer : GL_DEPTH_COMPONENT, GL_STENCIL_INDEX, or GL_DEPTH_STENCIL
	 */
//	void attachRenderbuffer(GLenum internalFormat);
	
	/**
	 * Attach a color texture to the Fbo.\n
	 * This function fails if the maximum number of attached color textures is already reached.\n
	 * When calling this function, no Fbo should be bound.\n
	 * @param  internalFormat  Internal format of the texture : GL_RGBA, GL_RGB, GL_RGBA32F, or GL_RGB32F
	 * @param  filter          Filter used to minify or magnify the texture (default value is GL_LINEAR)
	 * @return id of attachement
	 */
	int createAttachColorTexture(GLenum internalFormat, GLint filter = GL_LINEAR);

	/**
	 * Attach a color texture to the Fbo.\n
	 * This function fails if the maximum number of attached color textures is already reached.\n
	 * When calling this function, no Fbo should be bound.\n
	 * @param textureId id of texture to attach  (default is 0: texture us creatated with internal format)
	 * @return id of attachement
	 */
	int attachColorTexture(CGoGNGLuint textureId);

	/**
	 * Create and attach a depth texture to the Fbo.\n
	 * This function fails if a depth texture was already attached to the Fbo.\n
	 * When calling this function, no Fbo should be bound.\n
	 * @param  filter  Filter used to minify or magnify the texture (default value is GL_LINEAR)
	 * @return true if creation & attachment success
	 */
	bool createAttachDepthTexture(GLint filter = GL_LINEAR);

	/**
	 * Attach a depth texture to the Fbo.\n
	 * This function fails if a depth texture was already attached to the Fbo.\n
	 * When calling this function, no Fbo should be bound.\n
	 * @param textureId depth texture from another FBO, with getDepthTexId
	 * @return true if attachment success
	 */
	bool attachDepthTexture(CGoGNGLuint textureId);


	/**
	 * Enable every color attachments of the Fbo.\n
	 * Useless for Fbos containing only one color attachment.\n
	 * When calling this function, this Fbo must be bound.\n
	 */
	void enableAllColorAttachments();

	/**
	 * Enable a specific color attachment of the Fbo.\n
	 * Useless for Fbos containing only one color attachment.\n
	 * When calling this function, this Fbo must be bound.\n
	 * \param  num  Number of the color attachment that should be enabled
	 */
	void enableColorAttachment(int num);
	
	/**
	 * Get an attached color texture id.
	 * \param  att  Number of the color texture (the number is equivalent to the color attachment)
	 * \returns  Color texture id
	 */
	CGoGNGLuint getColorTexId(int att);
	
	/**
	 * Get the attached depth texture id.
	 * \returns  Depth texture id
	 */
	CGoGNGLuint getDepthTexId();

	/**
	 * draw full screen attached texture
	 * @param attach id of attachment
	 */
	void draw(int attach=0);

	/**
	 * draw full screen attached texture
	 * @param attach id of attachment
	 */
	void drawWithDepth(int attach=0);

	/**
	 * draw full screen attached texture
	 * @param attach id of attachment
	 */
	void drawWithDepth(int attach, CGoGNGLuint depthTexId);

	/**
	 *
	 */
	static void draw(Utils::GLSLShader* shader);

	/**
	 * Bind this Fbo.\n
	 * When calling this function, no Fbo should be bound.\n
	 */
	void bind();
	
	/**
	 * Unbind this Fbo.\n
	 */
	void unbind();

	/**
	 * Call glFlush(), unbind this Fbo, call glDrawBuffer(GL_BACK).\n
	 */
	void safeUnbind();

	/**
	 * Check the completeness of this Fbo.\n
	 * When calling this function, no Fbo should be bound.\n
	 */
	void checkFBO();

	/**
	 * \returns  Width of the Fbo in pixels.
	 */
	unsigned int getWidth() const;

	/**
	 * \returns  Height of the Fbo in pixels.
	 */
	unsigned int getHeight() const;

protected:

	/// Width (in pixels) of the Fbo.
	unsigned int m_width;
	
	/// Height (in pixels) of the Fbo.
	unsigned int m_height;

	/// Maximum number of color attachments.
	int m_maxColorAttachments;

	/// Fbo id.
	CGoGNGLuint m_fboId;
	
	/// Render buffer id.
	CGoGNGLuint m_renderBufferId;
	
	/// Color textures ids (up to m_maxColorAttachments color textures).
	std::vector<CGoGNGLuint> m_colorTexId;
	
	/// Depth textures ids (only one depth texture actually).
	std::vector<CGoGNGLuint> m_depthTexId;

	/// Color attachments of the Fbo.
	CGoGNGLenumTable m_colorAttachmentPoints;
	
	/// Original viewport (x, y, width, height), saved when binding the Fbo, restored when unbinding it.
	GLint m_oldViewport[4];
	
	/// Indicates wether the Fbo is bound or not.
	bool m_bound;
	
	/// Indicates wether any Fbo is bound or not.
	static bool sm_anyFboBound;
};

} // namespace Utils

} // namespace CGoGN

#endif	/* FRAMEBUFFER_HPP */
