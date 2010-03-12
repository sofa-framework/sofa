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

#ifdef SOFA_HAVE_GLEW
#define _GLEW(x) (x)
#define _GLEWF(x) x
#else
#define _GLEW(x) 0
#define _GLEWF(x)
#endif

namespace sofa
{
namespace helper
{
namespace gl
{
static unsigned int targetTable[io::Image::COUNT_OF_TEXTURE_TYPES] =
{
    GL_TEXTURE_2D,
    _GLEW(GL_TEXTURE_3D),
    _GLEW(GL_TEXTURE_CUBE_MAP)
};

static unsigned int typeTable[io::Image::COUNT_OF_DATA_TYPES] =
{
    GL_UNSIGNED_BYTE,     // UNORM8
    GL_UNSIGNED_SHORT,    // UNORM16
    GL_UNSIGNED_INT,      // UINT32
    _GLEW(GL_HALF_FLOAT), // HALF
    GL_FLOAT,             // FLOAT
    GL_UNSIGNED_BYTE      // UCOMPRESSED
};

static unsigned int formatTable[io::Image::COUNT_OF_CHANNEL_FORMATS] =
{
    GL_LUMINANCE,       // L
    GL_LUMINANCE_ALPHA, // LA
    GL_RED,             // R
    _GLEW(GL_RG),       // RG
    GL_RGB,             // RGB
    GL_RGBA,            // RGBA
    GL_BGR,             // BGR
    GL_BGRA             // BGRA
};

static unsigned int internalFormatTable[io::Image::COUNT_OF_DATA_TYPES][io::Image::COUNT_OF_CHANNEL_FORMATS] =
{
    // UNORM8
    {
        GL_LUMINANCE8,          // L
        GL_LUMINANCE8_ALPHA8,   // LA
        _GLEW(GL_R8),           // R
        _GLEW(GL_RG8),          // RG
        GL_RGB8,                // RGB
        GL_RGBA8,               // RGBA
        GL_RGB8,                // BGR
        GL_RGBA8                // BGRA
    },
    // UNORM16
    {
        GL_LUMINANCE16,         // L
        GL_LUMINANCE16_ALPHA16, // LA
        _GLEW(GL_R16),          // R
        _GLEW(GL_RG16),         // RG
        GL_RGB16,               // RGB
        GL_RGBA16,              // RGBA
        GL_RGB16,               // BGR
        GL_RGBA16               // BGRA
    },
    // UINT32
    {
        _GLEW(GL_LUMINANCE32UI_EXT),       // L
        _GLEW(GL_LUMINANCE_ALPHA32UI_EXT), // LA
        _GLEW(GL_R32UI),                   // R
        _GLEW(GL_RG32UI),                  // RG
        _GLEW(GL_RGB32UI),                 // RGB
        _GLEW(GL_RGBA32UI),                // RGBA
        _GLEW(GL_RGB32UI),                 // BGR
        _GLEW(GL_RGBA32UI)                 // BGRA
    },
    // HALF
    {
        _GLEW(GL_LUMINANCE16F_ARB),        // L
        _GLEW(GL_LUMINANCE_ALPHA16F_ARB),  // LA
        _GLEW(GL_R16F),                    // R
        _GLEW(GL_RG16F),                   // RG
        _GLEW(GL_RGB16F),                  // RGB
        _GLEW(GL_RGBA16F),                 // RGBA
        _GLEW(GL_RGB16F),                  // BGR
        _GLEW(GL_RGBA16F)                  // BGRA
    },
    // FLOAT
    {
        _GLEW(GL_LUMINANCE32F_ARB),        // L
        _GLEW(GL_LUMINANCE_ALPHA32F_ARB),  // LA
        _GLEW(GL_R32F),                    // R
        _GLEW(GL_RG32F),                   // RG
        _GLEW(GL_RGB32F),                  // RGB
        _GLEW(GL_RGBA32F),                 // RGBA
        _GLEW(GL_RGB32F),                  // BGR
        _GLEW(GL_RGBA32F)                  // BGRA
    },
    // UCOMPRESSED
    {
        _GLEW(GL_COMPRESSED_LUMINANCE_LATC1_EXT),          // L
        _GLEW(GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT),    // LA
        _GLEW(GL_COMPRESSED_RED_RGTC1),                    // R
        _GLEW(GL_COMPRESSED_RG_RGTC2),                     // RG
        _GLEW(GL_COMPRESSED_RGB_S3TC_DXT1_EXT),            // RGB
        _GLEW(GL_COMPRESSED_RGBA_S3TC_DXT5_EXT),           // RGBA
        0,                                                 // BGR
        0                                                  // BGRA
    }
};

void Texture::init(void)
{
    io::Image::TextureType textureType = image->getTextureType();
    target = GL_TEXTURE_2D; // Default value in case the format is not supported.

    // Check OpenGL support.
    // TODO: Check for non-power-of-two textures support.
    switch (textureType)
    {
    case io::Image::TEXTURE_INVALID:
        std::cerr << "sofa::helper::gl::Texture::init: Invalid texture type." << std::endl;
        return;

    case io::Image::TEXTURE_3D:
#ifndef SOFA_HAVE_GLEW
        std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, the GLEW library is missing." << std::endl;
        return;
#endif
        if (!_GLEW(GLEW_VERSION_1_2) &&
            !_GLEW(GLEW_EXT_texture3D))
        {
            std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                    "either OpenGL 1.2 "
                    "or GL_EXT_texture3D must be supported." << std::endl;
            return;
        }
        break;

    case io::Image::TEXTURE_CUBE:
#ifndef SOFA_HAVE_GLEW
        std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, the GLEW library is missing." << std::endl;
        return;
#endif
        if (!_GLEW(GLEW_VERSION_1_3) &&
            !_GLEW(GLEW_ARB_texture_cube_map) &&
            !_GLEW(GLEW_EXT_texture_cube_map))
        {
            std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                    "either OpenGL 1.3 "
                    "or GL_ARB_texture_cube_map "
                    "or GL_EXT_texture_cube_map must be supported." << std::endl;
            return;
        }
        break;

    default:;
    }

    switch (image->getDataType())
    {
    case io::Image::UINT32:
#ifndef SOFA_HAVE_GLEW
        std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, the GLEW library is missing." << std::endl;
        return;
#endif
        if (image->getChannelFormat() <= io::Image::LA)
        {
            if (!_GLEW(GLEW_EXT_texture_integer))
            {
                std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                        "GL_EXT_texture_integer must be supported." << std::endl;
                return;
            }
        }
        else
        {
            if (!_GLEW(GLEW_VERSION_3_0) &&
                !_GLEW(GLEW_EXT_texture_integer))
            {
                std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                        "either OpenGL 3.0 "
                        "or GL_EXT_texture_integer must be supported." << std::endl;
                return;
            }
        }
        break;

    case io::Image::HALF:
#ifndef SOFA_HAVE_GLEW
        std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, the GLEW library is missing." << std::endl;
        return;
#endif
        if (!_GLEW(GLEW_VERSION_3_0) &&
            !_GLEW(GLEW_ARB_half_float_pixel))
        {
            std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                    "either OpenGL 3.0 "
                    "or GL_ARB_half_float_pixel must be supported." << std::endl;
            return;
        }
        /* Pass through. */

    case io::Image::FLOAT:
#ifndef SOFA_HAVE_GLEW
        std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, the GLEW library is missing." << std::endl;
        return;
#endif
        if (!_GLEW(GLEW_VERSION_3_0) &&
            !_GLEW(GLEW_ARB_texture_float) &&
            !_GLEW(GLEW_ATI_texture_float))
        {
            std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                    "either OpenGL 3.0 "
                    "or GL_ARB_texture_float "
                    "or GL_ATI_texture_float must be supported." << std::endl;
            return;
        }
        break;

    case io::Image::UCOMPRESSED:
#ifndef SOFA_HAVE_GLEW
        std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, the GLEW library is missing." << std::endl;
        return;
#endif
        switch (image->getChannelFormat())
        {
        case io::Image::L:
        case io::Image::LA:
            if (!_GLEW(GLEW_EXT_texture_compression_latc))
            {
                std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                        "GL_EXT_texture_compression_latc must be supported." << std::endl;
                return;
            }
            break;

        case io::Image::R:
        case io::Image::RG:
            if (!_GLEW(GLEW_VERSION_3_0) &&
                !_GLEW(GLEW_ARB_texture_compression_rgtc) &&
                !_GLEW(GLEW_EXT_texture_compression_rgtc))
            {
                std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                        "either OpenGL 3.0 "
                        "or GL_ARB_texture_compression_rgtc "
                        "or GL_EXT_texture_compression_rgtc must be supported." << std::endl;
                return;
            }
            break;

        case io::Image::RGB:
        case io::Image::RGBA:
            if (!_GLEW(GLEW_EXT_texture_compression_s3tc))
            {
                std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                        "GL_EXT_texture_compression_s3tc must be supported." << std::endl;
                return;
            }
            break;
        default:;
        }
        break;

    default:;
        // Always supported.
    }

    target = targetTable[textureType];
    unsigned internalFormat = internalFormatTable[image->getDataType()][image->getChannelFormat()];
    unsigned format = formatTable[image->getChannelFormat()];
    unsigned type = typeTable[image->getDataType()];
    unsigned mipmaps = image->getMipmapCount();

    glGenTextures(1, &id); // Create the texture.
    glBindTexture(target, id);

    switch (textureType)
    {
    case io::Image::TEXTURE_2D:
        if (image->getDataType() == io::Image::UCOMPRESSED)
            for (unsigned i = 0; i < mipmaps; i++)
                _GLEWF(glCompressedTexImage2D(target, i, internalFormat, image->getWidth(i), image->getHeight(i), 0,
                        image->getMipmapSize(i), image->getMipmapPixels(i)));
        else
            for (unsigned i = 0; i < mipmaps; i++)
                glTexImage2D(target, i, internalFormat, image->getWidth(i), image->getHeight(i), 0,
                        format, type, image->getMipmapPixels(i));
        break;

    case io::Image::TEXTURE_3D:
        // 3D textura
        if (image->getDataType() == io::Image::UCOMPRESSED)
            for (unsigned i = 0; i < mipmaps; i++)
                _GLEWF(glCompressedTexImage3D(target, i, internalFormat, image->getWidth(i), image->getHeight(i),
                        image->getDepth(i), 0, image->getMipmapSize(i), image->getMipmapPixels(i)));
        else
            for (unsigned i = 0; i < mipmaps; i++)
                _GLEWF(glTexImage3D(target, i, internalFormat, image->getWidth(i), image->getHeight(i),
                        image->getDepth(i), 0, format, type, image->getMipmapPixels(i)));
        break;

    case io::Image::TEXTURE_CUBE:
        // Cubemapa
        if (image->getDataType() == io::Image::UCOMPRESSED)
            for (unsigned j = 0; j < 6; j++)
                for (unsigned i = 0; i < mipmaps; i++)
                    _GLEWF(glCompressedTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + j, i, internalFormat,
                            image->getWidth(i), image->getHeight(i), 0, image->getMipmapSize(i) / 6,
                            (image->getPixels())? image->getCubeMipmapPixels(j, i) : 0));
        else
            for (unsigned j = 0; j < 6; j++)
                for (unsigned i = 0; i < mipmaps; i++)
                    _GLEWF(glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + j, i, internalFormat,
                            image->getWidth(i), image->getHeight(i), 0, format, type,
                            (image->getPixels())? image->getCubeMipmapPixels(j, i) : 0));
        break;

    default:;
    }

    if (linearInterpolation)
    {
        if (mipmaps > 1)
            glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        else
            glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        if (_GLEW(GLEW_EXT_texture_filter_anisotropic))
        {
            int maxAniso;
            glGetIntegerv(_GLEW(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT), &maxAniso);
            glTexParameteri(target, _GLEW(GL_TEXTURE_MAX_ANISOTROPY_EXT), maxAniso);
        }
    }
    else
    {
        if (mipmaps > 1)
            glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
        else
            glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    if (repeat && textureType != io::Image::TEXTURE_CUBE)
    {
        glTexParameteri( target, GL_TEXTURE_WRAP_S, GL_REPEAT );
        glTexParameteri( target, GL_TEXTURE_WRAP_T, GL_REPEAT );
        glTexParameteri( target, GL_TEXTURE_WRAP_R, GL_REPEAT );
    }
    else
    {
        if (_GLEW(GLEW_VERSION_1_2) ||
            _GLEW(GLEW_EXT_texture_edge_clamp) ||
            _GLEW(GLEW_SGIS_texture_edge_clamp))
        {
            glTexParameteri( target, GL_TEXTURE_WRAP_S, _GLEW(GL_CLAMP_TO_EDGE) );
            glTexParameteri( target, GL_TEXTURE_WRAP_T, _GLEW(GL_CLAMP_TO_EDGE) );
            glTexParameteri( target, GL_TEXTURE_WRAP_R, _GLEW(GL_CLAMP_TO_EDGE) );
        }
        else
        {
            glTexParameteri( target, GL_TEXTURE_WRAP_S, GL_CLAMP );
            glTexParameteri( target, GL_TEXTURE_WRAP_T, GL_CLAMP );
            glTexParameteri( target, GL_TEXTURE_WRAP_R, GL_CLAMP );
        }
    }
    glBindTexture(target, 0);
}

void Texture::bind(void)
{
    glBindTexture(target, id);
}

void Texture::unbind(void)
{
    glBindTexture(target, 0);
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
