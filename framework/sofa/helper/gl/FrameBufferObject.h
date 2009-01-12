/*
 * FrameBufferObject.h
 *
 *  Created on: 6 janv. 2009
 *      Author: froy
 */

#ifndef FRAMEBUFFEROBJECT_H_
#define FRAMEBUFFEROBJECT_H_

#include <sofa/helper/gl/Texture.h>
#include <sofa/helper/helper.h>

namespace sofa
{
namespace helper
{
namespace gl
{

class SOFA_HELPER_API FrameBufferObject
{

private:
    unsigned int width, height;
    GLuint id;
    GLuint depthTexture, colorTexture;
    bool initialized;
public:
    FrameBufferObject();
    virtual ~FrameBufferObject();

    void init(unsigned int width, unsigned height);

    void start();
    void stop();

    GLuint getDepthTexture();
    GLuint getColorTexture();

};

} //gl

} //helper

} //sofa

#endif /* FRAMEBUFFEROBJECT_H_ */
