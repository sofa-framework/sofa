/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/SofaFramework.h>
#include <sofa/helper/gl/Texture.h>
#include <assert.h>
#include <stdio.h>
#include <sofa/helper/system/glu.h>

namespace sofa
{
namespace helper
{
namespace gl
{
static unsigned int targetTable[io::Image::COUNT_OF_TEXTURE_TYPES] =
{
    GL_TEXTURE_2D,
#if defined(GL_VERSION_1_2)
    GL_TEXTURE_3D,
#else
    0,
#endif
#if defined(GL_VERSION_1_3)
    GL_TEXTURE_CUBE_MAP
#endif
};

static unsigned int typeTable[io::Image::COUNT_OF_DATA_TYPES] =
{
    GL_UNSIGNED_BYTE,     // UNORM8
    GL_UNSIGNED_SHORT,    // UNORM16
    GL_UNSIGNED_INT,      // UINT32
#if defined(GL_VERSION_3_0)
    GL_HALF_FLOAT,        // HALF
#else
    0,                    // HALF
#endif
    GL_FLOAT,             // FLOAT
    GL_UNSIGNED_BYTE      // UCOMPRESSED
};

static unsigned int formatTable[io::Image::COUNT_OF_CHANNEL_FORMATS] =
{
    GL_LUMINANCE,       // L
    GL_LUMINANCE_ALPHA, // LA
    GL_RED,             // R
#if defined(GL_VERSION_3_0)
    GL_RG,              // RG
#else
    0,                  // RG
#endif
    GL_RGB,             // RGB
    GL_RGBA,            // RGBA
#if defined(GL_VERSION_1_2)
    GL_BGR,             // BGR
    GL_BGRA             // BGRA
#endif
};

static unsigned int internalFormatTable[io::Image::COUNT_OF_DATA_TYPES][io::Image::COUNT_OF_CHANNEL_FORMATS] =
{
    // UNORM8
    {
        GL_LUMINANCE8,          // L
        GL_LUMINANCE8_ALPHA8,   // LA
#if defined(GL_VERSION_3_0)
        GL_R8,                  // R
        GL_RG8,                 // RG
#else
        0, 0,
#endif
        GL_RGB8,                // RGB
        GL_RGBA8,               // RGBA
        GL_RGB8,                // BGR
        GL_RGBA8                // BGRA
    },
    // UNORM16
    {
        GL_LUMINANCE16,         // L
        GL_LUMINANCE16_ALPHA16, // LA
#if defined(GL_VERSION_3_0)
        GL_R16,          // R
        GL_RG16,         // RG
#else
        0, 0,
#endif
        GL_RGB16,               // RGB
        GL_RGBA16,              // RGBA
        GL_RGB16,               // BGR
        GL_RGBA16               // BGRA
    },
    // UINT32
    {
#if defined(GL_EXT_texture_integer)
        GL_LUMINANCE32UI_EXT,       // L
        GL_LUMINANCE_ALPHA32UI_EXT, // LA
#else
        0, 0,
#endif
#if defined(GL_VERSION_3_0)
        GL_R32UI,                   // R
        GL_RG32UI,                  // RG
        GL_RGB32UI,                 // RGB
        GL_RGBA32UI,                // RGBA
        GL_RGB32UI,                 // BGR
        GL_RGBA32UI                 // BGRA
#endif
    },
    // HALF
    {
#if defined(GL_ARB_texture_float)
        GL_LUMINANCE16F_ARB,        // L
        GL_LUMINANCE_ALPHA16F_ARB,  // LA
#else
        0, 0,
#endif
#if defined(GL_VERSION_3_0)
        GL_R16F,                    // R
        GL_RG16F,                   // RG
        GL_RGB16F,                  // RGB
        GL_RGBA16F,                 // RGBA
        GL_RGB16F,                  // BGR
        GL_RGBA16F                  // BGRA
#endif
    },
    // FLOAT
    {
#if defined(GL_ARB_texture_float)
        GL_LUMINANCE32F_ARB,        // L
        GL_LUMINANCE_ALPHA32F_ARB,  // LA
#else
        0, 0,
#endif
#if defined(GL_VERSION_3_0)
        GL_R32F,                    // R
        GL_RG32F,                   // RG
        GL_RGB32F,                  // RGB
        GL_RGBA32F,                 // RGBA
        GL_RGB32F,                  // BGR
        GL_RGBA32F                  // BGRA
#endif
    },
    // UCOMPRESSED
    {
#if defined(GL_EXT_texture_compression_latc)
        GL_COMPRESSED_LUMINANCE_LATC1_EXT,          // L
        GL_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT,    // LA
#else
        0, 0,
#endif
#if defined(GL_VERSION_3_0)
        GL_COMPRESSED_RED_RGTC1,                    // R
        GL_COMPRESSED_RG_RGTC2,                     // RG
#else
        0, 0,
#endif
#if defined(GL_EXT_texture_compression_s3tc)
        GL_COMPRESSED_RGB_S3TC_DXT1_EXT,            // RGB
        GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,           // RGBA
#endif
    }
};

static unsigned int internalFormatTableSRGB[io::Image::COUNT_OF_DATA_TYPES][io::Image::COUNT_OF_CHANNEL_FORMATS] =
{
#if defined(GL_EXT_texture_sRGB)
    // UNORM8
    {
        GL_SLUMINANCE8_EXT,          // L
        GL_SLUMINANCE8_ALPHA8_EXT,   // LA
        0, 0,                        // R, RG
        GL_SRGB8_EXT,                // RGB
        GL_SRGB8_ALPHA8_EXT,         // RGBA
        GL_SRGB8_EXT,                // BGR
        GL_SRGB8_ALPHA8_EXT          // BGRA
    },
#else
    { 0 },
#endif
    // UNORM16
    { 0 },
    // UINT32
    { 0 },
    // HALF
    { 0 },
    // FLOAT
    { 0 },
    // UCOMPRESSED
    {
        0, 0, 0, 0,                  // L, LA, R, RG
#if defined(GL_EXT_texture_sRGB)
        GL_COMPRESSED_SRGB_S3TC_DXT1_EXT,      // RGB
        GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT // RGBA
#endif
    }
};

static bool isPowerOfTwo(unsigned a)
{
    return a > 0 && !(a & (a - 1));
}






void Texture::update()
{
    io::Image::TextureType textureType = image->getTextureType();
    target = targetTable[textureType];
    unsigned format = formatTable[image->getChannelFormat()];
    unsigned type = typeTable[image->getDataType()];
    unsigned mipmaps = image->getMipmapCount();
    unsigned internalFormat = internalFormatTable[image->getDataType()][image->getChannelFormat()];

    if (srgbColorspace)
    {
        unsigned internalFormatSRGB = internalFormatTableSRGB[image->getDataType()][image->getChannelFormat()];
        if (internalFormatSRGB)
        {
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_EXT_texture_sRGB) && defined(GLEW_ARB_framebuffer_sRGB)
            if (GLEW_EXT_texture_sRGB && GLEW_ARB_framebuffer_sRGB)
                internalFormat = internalFormatSRGB;
            else
#endif
            {
                std::cerr << "sofa::helper::gl::Texture::init: SRGB colorspace is unsupported, "
                        "GLEW_EXT_texture_srgb or GLEW_ARB_framebuffer_sRGB is missing." << std::endl;
            }
        }
        else
        {
            std::cerr << "sofa::helper::gl::Texture::init: SRGB colorspace "
                    "isn't supported with the given texture format." << std::endl;
        }
    }

    glBindTexture(target, id);
    glPixelStorei(GL_PACK_ALIGNMENT,1);
    switch (textureType)
    {
    case io::Image::TEXTURE_2D:
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_1_3)
        if (image->getDataType() == io::Image::UCOMPRESSED)
            for (unsigned i = 0; i < mipmaps; i++)
                glCompressedTexImage2D(target, i, internalFormat, image->getWidth(i), image->getHeight(i), 0,
                        image->getMipmapSize(i), image->getMipmapPixels(i));
        else
#endif
            for (unsigned i = 0; i < mipmaps; i++)
                glTexImage2D(target, i, internalFormat, image->getWidth(i), image->getHeight(i), 0,
                        format, type, image->getMipmapPixels(i));
        break;

    case io::Image::TEXTURE_3D:
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_1_2)
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_1_3)
        if (image->getDataType() == io::Image::UCOMPRESSED)
            for (unsigned i = 0; i < mipmaps; i++)
                glCompressedTexImage3D(target, i, internalFormat, image->getWidth(i), image->getHeight(i),
                        image->getDepth(i), 0, image->getMipmapSize(i), image->getMipmapPixels(i));
        else
#endif
            for (unsigned i = 0; i < mipmaps; i++)
                glTexImage3D(target, i, internalFormat, image->getWidth(i), image->getHeight(i),
                        image->getDepth(i), 0, format, type, image->getMipmapPixels(i));
#endif
        break;

    case io::Image::TEXTURE_CUBE:
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_1_3)
        if (image->getDataType() == io::Image::UCOMPRESSED)
            for (unsigned j = 0; j < 6; j++)
                for (unsigned i = 0; i < mipmaps; i++)
                    glCompressedTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + j, i, internalFormat,
                            image->getWidth(i), image->getHeight(i), 0, image->getMipmapSize(i) / 6,
                            (image->getPixels())? image->getCubeMipmapPixels(j, i) : 0);
        else
            for (unsigned j = 0; j < 6; j++)
                for (unsigned i = 0; i < mipmaps; i++)
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + j, i, internalFormat,
                            image->getWidth(i), image->getHeight(i), 0, format, type,
                            (image->getPixels())? image->getCubeMipmapPixels(j, i) : 0);
#endif
        break;

    default:;
    }
    glPixelStorei(GL_PACK_ALIGNMENT, 4);

}


void Texture::init()
{
    io::Image::TextureType textureType = image->getTextureType();
    target = GL_TEXTURE_2D; // Default value in case the format is not supported.

    // Check OpenGL support.
    if (!isPowerOfTwo(image->getWidth()) ||
        !isPowerOfTwo(image->getHeight()) ||
        (image->getDepth() != 0 && !isPowerOfTwo(image->getDepth())))
    {
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_2_0)
        if (!GLEW_VERSION_2_0)
#endif
        {
            std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                    "NPOT textures are not supported." << std::endl;
            return;
        }
    }

    switch (textureType)
    {
    case io::Image::TEXTURE_INVALID:
        std::cerr << "sofa::helper::gl::Texture::init: Invalid texture type." << std::endl;
        return;

    case io::Image::TEXTURE_3D:
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_1_2)
        if (!GLEW_VERSION_1_2)
#endif
        {
            std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                    "OpenGL 1.2 is unsupported." << std::endl;
            return;
        }
        break;

    case io::Image::TEXTURE_CUBE:
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_1_3)
        if (!GLEW_VERSION_1_3)
#endif
        {
            std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                    "OpenGL 1.3 is unsupported." << std::endl;
            return;
        }
        break;

    default:;
    }

    switch (image->getDataType())
    {
    case io::Image::UINT32:
        if (image->getChannelFormat() <= io::Image::LA)
        {
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_EXT_texture_integer)
            if (!GLEW_EXT_texture_integer)
#endif
            {
                std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                        "GL_EXT_texture_integer is unsupported." << std::endl;
                return;
            }
        }
        else
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_3_0)
            if (!GLEW_VERSION_3_0)
#endif
            {
                std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                        "OpenGL 3.0 is unsupported." << std::endl;
                return;
            }
        break;

    case io::Image::HALF:
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_3_0)
        if (!GLEW_VERSION_3_0)
#endif
        {
            std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                    "OpenGL 3.0 is unsupported." << std::endl;
            return;
        }
        /* Pass through (no break!) */

    case io::Image::FLOAT:
        if (image->getChannelFormat() <= io::Image::LA)
        {
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_ARB_texture_float)
            if (!GLEW_ARB_texture_float)
#endif
            {
                std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                        "GLEW_ARB_texture_float is unsupported." << std::endl;
                return;
            }
        }
        else
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_3_0)
            if (!GLEW_VERSION_3_0)
#endif
            {
                std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                        "OpenGL 3.0 is unsupported." << std::endl;
                return;
            }
        break;

    case io::Image::UCOMPRESSED:
        switch (image->getChannelFormat())
        {
        case io::Image::L:
        case io::Image::LA:
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_EXT_texture_compression_latc)
            if (!GLEW_EXT_texture_compression_latc)
#endif
            {
                std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                        "GL_EXT_texture_compression_latc is unsupported." << std::endl;
                return;
            }
            break;

        case io::Image::R:
        case io::Image::RG:
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_3_0)
            if (!GLEW_VERSION_3_0)
#endif
            {
                std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                        "OpenGL 3.0 is unsupported." << std::endl;
                return;
            }
            break;

        case io::Image::RGB:
        case io::Image::RGBA:
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_EXT_texture_compression_s3tc)
            if (!GLEW_EXT_texture_compression_s3tc)
#endif
            {
                std::cerr << "sofa::helper::gl::Texture::init: Cannot load a texture, "
                        "GL_EXT_texture_compression_s3tc is unsupported." << std::endl;
                return;
            }
            break;
        default:;
        }
        break;

    default:;
        // Always supported.
    }

    unsigned mipmaps = image->getMipmapCount();

    glGenTextures(1, &id); // Create the texture.
    update();


#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_1_4)
    if (GLEW_VERSION_1_4 && generateMipmaps)
        glTexParameteri(target, GL_GENERATE_MIPMAP, GL_TRUE);
    else
#endif
        generateMipmaps = false;

    if (linearInterpolation)
    {
        if (generateMipmaps || mipmaps > 1)
            glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        else
            glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

#if defined(SOFA_HAVE_GLEW) && defined(GLEW_EXT_texture_filter_anisotropic)
        if (GLEW_EXT_texture_filter_anisotropic)
        {
            GLint maxAniso;
            glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
            glTexParameteri(target, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAniso);
        }
#endif
    }
    else
    {
        if (generateMipmaps || mipmaps > 1)
            glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
        else
            glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    if (repeat && textureType != io::Image::TEXTURE_CUBE)
    {
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_1_2)
        glTexParameteri( target, GL_TEXTURE_WRAP_S, GL_REPEAT );
        glTexParameteri( target, GL_TEXTURE_WRAP_T, GL_REPEAT );
        glTexParameteri( target, GL_TEXTURE_WRAP_R, GL_REPEAT );
#else
        std::cerr << __FUNCTION__<< " GLEW_VERSION_1_2 required for cubic texture." << std::endl;
#endif
    }
    else
    {
#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_1_2)
            glTexParameteri( target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
            glTexParameteri( target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
            glTexParameteri( target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
#else
        std::cerr << __FUNCTION__<< " GLEW_VERSION_1_2 required for cubic texture." << std::endl;
#endif
    }

#if defined(SOFA_HAVE_GLEW) && defined(GLEW_ARB_seamless_cube_map)
    // This is a global state so probably should be moved to a more appropriate location.
    if (textureType == io::Image::TEXTURE_CUBE)
        if (GLEW_ARB_seamless_cube_map)
            glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
#endif

#if defined(SOFA_HAVE_GLEW) && defined(GLEW_VERSION_1_2)
    if ((generateMipmaps || mipmaps > 1) && GLEW_VERSION_1_2)
    {
        glTexParameterf(target, GL_TEXTURE_MIN_LOD, minLod);
        glTexParameterf(target, GL_TEXTURE_MAX_LOD, maxLod);
    }
#endif
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
