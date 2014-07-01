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
#ifndef __SNAPSHOT_HD__
#define __SNAPSHOT_HD__

#include <QImage>
#include <GL/glew.h>

namespace CGoGN
{

namespace Utils
{

/**
 * HD snapshot class
 * Use OGL FBO to do squared snapshot until 16k*16k
 */
class SnapshotHD
{
protected:
	int imgSz;
	GLuint color_tex;
	GLuint fb;
	GLuint depth_rb;
	unsigned char* pixels;
	GLint old_viewport[4];

public:
	/**
	* constructor
	* @param sz size of rendering (in a square sz*sz) must be < 16384
	*/
	inline SnapshotHD(int sz);
	/**
	 * @brief destructor
	 */
	inline ~SnapshotHD();
	/**
	 * @brief bind the fbo for snapshot (call before updateGL();)
	 */
	inline void bind();

	/**
	 * @brief unbind the fbo
	 */
	inline void unbind();

	/**
	 * @brief save the screen to file
	 * @param name file name (use anything supported by Qt)
	 */
	inline void snapshot(const std::string& name);
};	

SnapshotHD::SnapshotHD(int sz):
 imgSz(sz)
{
	glGenTextures(1, &color_tex);
	glBindTexture(GL_TEXTURE_2D, color_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, imgSz, imgSz, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glGenFramebuffersEXT(1, &fb);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, color_tex, 0);
	glGenRenderbuffersEXT(1, &depth_rb);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depth_rb);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, imgSz, imgSz);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth_rb);

	pixels = new unsigned char[imgSz*imgSz*3];
}


SnapshotHD::~SnapshotHD()
{
	if (pixels != NULL)
		delete pixels;
	glDeleteTextures(1, &color_tex);
	glDeleteRenderbuffersEXT(1, &depth_rb);
	glDeleteFramebuffersEXT(1, &fb);
}

void SnapshotHD::snapshot(const std::string& name)
{
	for(int i=0; i<imgSz; ++i)
		glReadPixels(0, imgSz-1-i, imgSz, 1, GL_RGB, GL_UNSIGNED_BYTE, pixels+i*imgSz*3);

	QImage qim(pixels,imgSz,imgSz,QImage::Format_RGB888);
	qim.save(QString(name.c_str()));
}

void SnapshotHD::bind()
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);
	glGetIntegerv( GL_VIEWPORT, old_viewport );

	glViewport(0,0,imgSz,imgSz);
}

void SnapshotHD::unbind()
{
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glViewport(old_viewport[0],old_viewport[1],old_viewport[2],old_viewport[3]);
}

}
}

#endif

