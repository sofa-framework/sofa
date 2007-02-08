#ifndef SOFA_HELPER_GL_TEXTURE_H
#define SOFA_HELPER_GL_TEXTURE_H

#ifdef _WIN32
#include <windows.h>
#endif // _WIN32

#include <GL/gl.h>

#include <sofa/helper/io/Image.h>

namespace sofa
{

namespace helper
{

namespace gl
{

//using namespace sofa::defaulttype;

class Texture
{
private:
    io::Image *image;
    GLuint id;
public:
    Texture (io::Image *img):image(img),id(0) {};
    io::Image* getImage(void);
    void   bind(void);
    void   unbind(void);
    void   init (void);
    ~Texture();
};

} // namespace gl

} // namespace helper

} // namespace sofa

#endif
