/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 * FrameBufferObject.cpp
 *
 *  Created on: 6 janv. 2009
 *      Author: froy
 */
#include <cassert>
#include <sofa/helper/gl/FrameBufferObject.h>


namespace sofa
{

namespace helper
{

namespace gl
{

FrameBufferObject::FrameBufferObject(bool depthTexture, bool enableDepth, bool enableColor)
    :width(0)
    ,height(0)
    ,depthTextureID(0)
    ,colorTextureID(0)
    ,initialized(false)
    ,depthTexture(depthTexture)
    ,enableDepth(enableDepth)
    ,enableColor(enableColor)
{

}

FrameBufferObject::FrameBufferObject(const fboParameters& fboParams, bool depthTexture, bool enableDepth, bool enableColor)
    :width(0)
    ,height(0)
    ,depthTextureID(0)
    ,colorTextureID(0)
    ,initialized(false)
    ,_fboParams(fboParams)
    ,depthTexture(depthTexture)
    ,enableDepth(enableDepth)
    ,enableColor(enableColor)
{
}


FrameBufferObject::~FrameBufferObject()
{
    destroy();
}

void FrameBufferObject::destroy()
{
    if(initialized)
    {
        if(enableDepth)
        {
            if(depthTexture)
                glDeleteTextures( 1, &depthTextureID );
            else
                glDeleteRenderbuffersEXT(1, &depthTextureID);
        }

        if(enableColor)
        {
            glDeleteTextures( 1, &colorTextureID );
        }

        glDeleteFramebuffersEXT( 1, &id );
        initialized = false;
    }
}

bool FrameBufferObject::checkFBO()
{

    GLenum status;
    status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    switch(status)
    {
    case GL_FRAMEBUFFER_COMPLETE_EXT:
        return true;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
        assert(false && "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT");
        return false;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
        assert(false && "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT");
        return false;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
        assert(false && "GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT");
        return false;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
        assert(false && "GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT");
        return false;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
        assert(false && "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT");
        return false;
        break;
    case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
        assert(false && "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT");
        return false;
        break;
    case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
        assert(false && "GL_FRAMEBUFFER_UNSUPPORTED_EXT");
        return false;
        break;
    default:
        assert(false && "Unknown ERROR");
        return false;
    }


}

void FrameBufferObject::init(unsigned int width, unsigned height)
{
    if (!initialized)
    {
        this->width = width;
        this->height = height;
        glGenFramebuffersEXT(1, &id);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, id);

        if(enableDepth)
        {
            createDepthBuffer();
            initDepthBuffer();

            //choice between rendering depth into a texture or a renderbuffer
            if(depthTexture)
                glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, depthTextureID, 0);
            else
                glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthTextureID);
        }

        if(enableColor)
        {
            createColorBuffer();
            initColorBuffer();
            glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, colorTextureID, 0);
        }

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

        if(enableColor)
        {
            glDrawBuffer(GL_BACK);
            glReadBuffer(GL_BACK);;
        }

#ifdef _DEBUG
        checkFBO();
#endif
        initialized=true;
    }
    else
        setSize(width, height);
}


void FrameBufferObject::reinit(unsigned int width, unsigned height, bool lDepthTexture, bool lEnableDepth, bool lEnableColor )
{
    destroy();

    depthTexture = lDepthTexture;
    enableDepth = lEnableDepth;
    enableColor = lEnableColor;

    init(width, height);

}


void FrameBufferObject::start()
{
    if (initialized)
    {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, id);

        if(enableColor)
        {
            glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
            glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
        }
    }
}

void FrameBufferObject::stop()
{
    if (initialized)
    {
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

        if(enableColor)
        {
            glDrawBuffer(GL_BACK);
            glReadBuffer(GL_BACK);
        }
    }
}

GLuint FrameBufferObject::getID()
{
    return id;
}

GLuint FrameBufferObject::getDepthTexture()
{
    return depthTextureID;
}

GLuint FrameBufferObject::getColorTexture()
{
    return colorTextureID;
}

void FrameBufferObject::setSize(unsigned int width, unsigned height)
{
    if (initialized)
    {
        this->width = width;
        this->height = height;

        if(enableDepth)
            initDepthBuffer();
        if(enableColor)
            initColorBuffer();
    }
}

void FrameBufferObject::createDepthBuffer()
{
    //Depth Texture
    glEnable(GL_TEXTURE_2D);
    if(depthTexture)
        glGenTextures(1, &depthTextureID);
    else
        glGenRenderbuffersEXT(1, &depthTextureID);

}

void FrameBufferObject::createColorBuffer()
{
    //Color Texture
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &colorTextureID);
}

void FrameBufferObject::initDepthBuffer()
{
    if(depthTexture)
    {
        glBindTexture(GL_TEXTURE_2D, depthTextureID);
        glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

        glTexImage2D(GL_TEXTURE_2D, 0, _fboParams.depthInternalformat , width, height, 0,GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    else
    {
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthTextureID);
        glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT, width, height);
        glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);

    }
}

void FrameBufferObject::initColorBuffer()
{
    glBindTexture(GL_TEXTURE_2D, colorTextureID);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

    glTexImage2D(GL_TEXTURE_2D, 0, _fboParams.colorInternalformat,  width, height, 0, _fboParams.colorFormat, _fboParams.colorType, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

}


} //namespace gl

} //namespace helper

} //namespace sofa
