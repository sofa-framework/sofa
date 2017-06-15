/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 * FrameBufferObject.h
 *
 *  Created on: 6 janv. 2009
 *      Author: froy
 */

#ifndef FRAMEBUFFEROBJECT_H_
#define FRAMEBUFFEROBJECT_H_

#include <sofa/helper/helper.h>

#ifdef SOFA_HAVE_GLEW

#include <sofa/helper/helper.h>
#include <sofa/helper/system/gl.h>


namespace sofa
{
namespace helper
{
namespace gl
{

struct SOFA_HELPER_API fboParameters
{
    GLint  depthInternalformat; // GL_DEPTHCOMPONENT16 GL_DEPTHCOMPONENT24...
    GLint  colorInternalformat; // GL_RGB8, GL_RGB16...
    GLenum colorFormat; // GL_RGB, GL_RGBA, GL_BGR...
    GLenum colorType; // GL_UNSIGNED_BYTE GL_UNSIGNED_INT...

    fboParameters()
    {
        depthInternalformat = GL_DEPTH_COMPONENT24;
        colorInternalformat = GL_RGBA8;
        colorFormat = GL_RGBA;
        colorType = GL_UNSIGNED_BYTE;
    }
};

class SOFA_HELPER_API FrameBufferObject
{
private:
    GLint m_defaultWindowFramebufferID;

    unsigned int width, height;
    GLuint id;
    GLuint depthTextureID, colorTextureID;
    bool initialized;
    fboParameters _fboParams;
    bool depthTexture;
    bool enableDepth;
    bool enableColor;
    bool enableMipMap;

    static GLint getCurrentFramebufferID();

public:
    FrameBufferObject(bool depthTexture = false, bool enableDepth = true, bool enableColor = true, bool enableMipMap = false, GLint defaultWindowFramebuffer = getCurrentFramebufferID());
    virtual ~FrameBufferObject();

    FrameBufferObject(const fboParameters& FboFormat, bool depthTexture = false, bool enableDepth = true, bool enableColor = true, bool enableMipMap = false, GLint defaultWindowFramebuffer = getCurrentFramebufferID() );
    void setFormat(const fboParameters& fboParams) { _fboParams = fboParams; }

    void init(unsigned int width, unsigned height);
    void destroy();
    void reinit(unsigned int width, unsigned height, bool lDepthTexture, bool lEnableDepth, bool lEnableColor );

    void start();
    void stop();

    bool checkFBO();

    void setSize(unsigned int width, unsigned height);

    GLuint getID();
    GLuint getDepthTexture();
    GLuint getColorTexture();

    void createDepthBuffer();
    void createColorBuffer();
    void initDepthBuffer();
    void initColorBuffer();
};

} //gl

} //helper

} //sofa

#endif /* SOFA_HAVE_GLEW */

#endif /* FRAMEBUFFEROBJECT_H_ */
