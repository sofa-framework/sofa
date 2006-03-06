#include "Texture.h"

namespace Sofa
{

namespace Components
{

namespace GL
{

void Texture::init(void)
{
    glGenTextures(1, &id);						// Create The Texture
    std::cout << "Create Texture"<<std::endl;
    // Typical Texture Generation Using Data From The Bitmap
    glBindTexture(GL_TEXTURE_2D, id);
    glTexImage2D(GL_TEXTURE_2D, 0, 3, image->getWidth(), image->getHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, image->getData());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void Texture::bind(void)
{
    glBindTexture(GL_TEXTURE_2D, id);
}

void Texture::unbind(void)
{
    glBindTexture(GL_TEXTURE_2D, 0);
}

Texture::~Texture(void)
{
    glDeleteTextures(1, &id);
    delete image;
}

} // namespace GL

} // namespace Components

} // namespace Sofa
