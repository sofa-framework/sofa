#ifndef SOFA_COMPONENTS_GL_TEXTURE_H
#define SOFA_COMPONENTS_GL_TEXTURE_H

#ifdef _WIN32
#include <windows.h>
#endif // _WIN32

#include <GL/gl.h>

#include "../Common/Image.h"

namespace Sofa
{

namespace Components
{

namespace GL
{

using namespace Common;

class Texture
{
private:
    Image *image;
    GLuint id;
public:
    Texture (Image *img):image(img),id(0) {};
    Image* getImage(void);
    void   bind(void);
    void   unbind(void);
    void   init (void);
    ~Texture();
};

} // namespace GL

} // namespace Components

} // namespace Sofa

#endif
