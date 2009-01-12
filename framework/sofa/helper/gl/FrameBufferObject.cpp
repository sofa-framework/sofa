/*
 * FrameBufferObject.cpp
 *
 *  Created on: 6 janv. 2009
 *      Author: froy
 */

#include <sofa/helper/gl/FrameBufferObject.h>

namespace sofa
{
namespace helper
{
namespace gl
{

FrameBufferObject::FrameBufferObject()
    :width(0)
    ,height(0)
    ,depthTexture(0)
    ,initialized(false)
{

}

FrameBufferObject::~FrameBufferObject()
{

}

void FrameBufferObject::init(unsigned int width, unsigned height)
{
    if (!initialized)
    {
        this->width = width;
        this->height = height;

        glGenFramebuffersEXT(1, &id);
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, id);

        //Depth Texture
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &depthTexture);
        glBindTexture(GL_TEXTURE_2D, depthTexture);

        glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );

        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, width, height, 0,GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL);

        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_2D, depthTexture, 0);

        glBindTexture(GL_TEXTURE_2D, 0);

        //Color Texture
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &colorTexture);
        glBindTexture(GL_TEXTURE_2D, colorTexture);

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,  width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, colorTexture, 0);

        glDrawBuffer(GL_NONE);

        //debug
        //if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT) == GL_FRAMEBUFFER_COMPLETE_EXT)
        //	std::cout << "FBO OK" << std::endl;


        //glReadBuffer(GL_NONE);

        /*
        switch(glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT))
        {
        case GL_FRAMEBUFFER_COMPLETE_EXT:
        	std::cout << "Status: Framebuffer Initialisation OK"
        	<< std::endl;
        	break;
        case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
        	std::cout << "Error: Framebuffer Configuration"
        	<< " Unsupported" << std::endl;
        	break;
        default:
        	std::cout << "Error: Unknown Framebuffer"
        	<< " Configuration Error" << std::endl;
        	break;
        }
        */

        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);


        initialized=true;
    }

}

void FrameBufferObject::start()
{
    if (initialized)
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, id);
}

void FrameBufferObject::stop()
{
    if (initialized)
        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

GLuint FrameBufferObject::getDepthTexture()
{
    return depthTexture;
}

GLuint FrameBufferObject::getColorTexture()
{
    return colorTexture;
}

} //gl

} //helper

} //sofa
