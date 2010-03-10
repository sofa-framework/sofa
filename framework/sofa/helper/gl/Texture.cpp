/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/gl/Texture.h>
#include <assert.h>
#include <stdio.h>

namespace sofa
{

namespace helper
{

namespace gl
{

void Texture::init(void)
{
    if (image->getDataType() != io::Image::UINT8)
    {
        std::cerr << "sofa::helper::gl::Texture::init: DataType other than UINT8 hasn't been implemented yet." << std::endl;
        return;
    }
    if (image->getChannelFormat() != io::Image::L &&
        image->getChannelFormat() != io::Image::LA &&
        image->getChannelFormat() != io::Image::RGB &&
        image->getChannelFormat() != io::Image::RGBA)
    {
        std::cerr << "sofa::helper::gl::Texture::init: ChannelFormat other than L, RGB, and RGBA hasn't been implemented yet." << std::endl;
        return;
    }

    glGenTextures(1, &id); // Create The Texture
//     std::cout << "Create "<<image->getWidth()<<"x"<<image->getHeight()<<" Texture "<<id<<std::endl;
    // Typical Texture Generation Using Data From The Bitmap
    glBindTexture(GL_TEXTURE_2D, id);
    switch(image->getChannelFormat())
    {
    case io::Image::RGBA:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, image->getWidth(), image->getHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, image->getPixels());
        break;
    case io::Image::RGB:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, image->getWidth(), image->getHeight(), 0, GL_RGB, GL_UNSIGNED_BYTE, image->getPixels());
        break;
    case io::Image::LA:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8_ALPHA8, image->getWidth(), image->getHeight(), 0, GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, image->getPixels());
        break;
    case io::Image::L:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, image->getWidth(), image->getHeight(), 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, image->getPixels());
        break;
    default:;
    }
    if (linearInterpolation)
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    if (repeat)
    {
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_REPEAT );

    }
    else
    {
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP );
    }
//     std::cout << "Texture "<<id<<" Created"<<std::endl;
}

void Texture::bind(void)
{
    glBindTexture(GL_TEXTURE_2D, id);
}

void Texture::unbind(void)
{
    glBindTexture(GL_TEXTURE_2D, 0);
}

io::Image* Texture::getImage(void)
{
    return image;
}
Texture::~Texture(void)
{
    glDeleteTextures(1, &id);
    delete image;
}

} // namespace gl

} // namespace helper

} // namespace sofa

