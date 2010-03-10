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
#include <sofa/helper/system/config.h>
#include <sofa/helper/io/Image.h>
#include <sofa/helper/Factory.inl>
#include <stdio.h>

namespace sofa
{

namespace helper
{

template class SOFA_HELPER_API Factory<std::string, sofa::helper::io::Image, std::string>;

namespace io
{

SOFA_LINK_CLASS(ImageBMP)
SOFA_LINK_CLASS(ImagePNG)

const char *Image::strFromDataType[COUNT_OF_DATA_TYPES+1] =
{
    "UINT8",
    "UINT16",
    "UINT32",
    "HALF",
    "FLOAT",

    "COMPRESSED_DXT1",
    "COMPRESSED_DXT3",
    "COMPRESSED_DXT5",
    "COMPRESSED_L",
    "COMPRESSED_LA",

    "COUNT_OF_DATA_TYPES"
};

const char *Image::strFromChannelFormat[COUNT_OF_CHANNEL_FORMATS+1] =
{
    "L",
    "LA",
    "R",
    "RG",
    "RGB",
    "RGBA",
    "BGR",
    "BGRA",
    "COMPRESSED",

    "COUNT_OF_CHANNEL_FORMATS"
};

const char *Image::strFromTextureType[TEXTURE_INVALID+1] =
{
    "TEXTURE_2D",
    "TEXTURE_3D",
    "TEXTURE_CUBE",

    "TEXTURE_INVALID"
};

Image::Image()
    : data(NULL)
{
}

Image::~Image()
{
    clear();
}

unsigned Image::getWidth(unsigned mipmap) const
{
    unsigned result = width >> mipmap;
    return result ? result : 1;
}

unsigned Image::getHeight(unsigned mipmap) const
{
    unsigned result = height >> mipmap;
    return result ? result : 1;
}

unsigned Image::getDepth(unsigned mipmap) const
{
    unsigned result = depth >> mipmap;
    return result ? result : 1;
}

unsigned Image::getBytesPerPixel() const
{
    static unsigned table[COUNT_OF_DATA_TYPES][COUNT_OF_CHANNEL_FORMATS] =
    {
        // UINT8
        {
            1,  // L
            2,  // AL
            1,  // R
            2,  // RG
            3,  // RGB
            4,  // RGBA
            3,  // BGR
            4,  // BGRA
            0   // COMPRESSED
        },
        // UINT16
        {
            2,  // L
            4,  // AL
            2,  // R
            4,  // RG
            6,  // RGB
            8,  // RGBA
            6,  // BGR
            8,  // BGRA
            0   // COMPRESSED
        },
        // UINT32
        {
            4,  // L
            8,  // AL
            4,  // R
            8,  // RG
            12, // RGB
            16, // RGBA
            12, // BGR
            16, // BGRA
            0   // COMPRESSED
        },
        // HALF
        {
            2,  // L
            4,  // AL
            2,  // R
            4,  // RG
            6,  // RGB
            8,  // RGBA
            6,  // BGR
            8,  // BGRA
            0   // COMPRESSED
        },
        // FLOAT
        {
            4,  // L
            8,  // AL
            4,  // R
            8,  // RG
            12, // RGB
            16, // RGBA
            12, // BGR
            16, // BGRA
            0   // COMPRESSED
        },
        // COMPRESSED_DXT1
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        // COMPRESSED_DXT3
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        // COMPRESSED_DXT5
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        // COMPRESSED_L
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        // COMPRESSED_LA
        { 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };

    return table[dataType][channelFormat];
}

unsigned Image::getBytesPerBlock() const
{
    static unsigned table[COUNT_OF_DATA_TYPES] =
    {
        0,  // UINT8
        0,  // UINT16
        0,  // UINT32
        0,  // HALF
        0,  // FLOAT
        8,  // COMPRESSED_DXT1
        16, // COMPRESSED_DXT3
        16, // COMPRESSED_DXT5
        8,  // COMPRESSED_L
        16  // COMPRESSED_LA
    };

    return table[dataType];
}

unsigned Image::getBytesPerChannel() const
{
    return getBytesPerPixel() / getChannelCount();
}

unsigned Image::getChannelCount() const
{
    static unsigned table[COUNT_OF_CHANNEL_FORMATS] =
    {
        1,  // L
        2,  // AL
        1,  // R
        2,  // RG
        3,  // RGB
        4,  // RGBA
        3,  // BGR
        4,  // BGRA
        0   // COMPRESSED
    };

    static unsigned compTable[COUNT_OF_DATA_TYPES] =
    {
        0,  // UINT8
        0,  // UINT16
        0,  // UINT32
        0,  // HALF
        0,  // FLOAT
        3,  // COMPRESSED_DXT1
        4,  // COMPRESSED_DXT3
        4,  // COMPRESSED_DXT5
        1,  // COMPRESSED_L
        2   // COMPRESSED_LA
    };

    return (channelFormat == COMPRESSED)? compTable[dataType] : table[channelFormat];
}

unsigned Image::getMipmapCount() const
{
    return mipmaps;
}

unsigned Image::getPixelCount() const
{
    return getImageSize() / getBytesPerPixel();
}

unsigned Image::getLineSize(unsigned mipmap) const
{
    return getWidth(mipmap) * getBytesPerPixel();
}

unsigned Image::getMipmapSize(unsigned mipmap) const
{
    // Return the size of one mipmap in bytes. For cubemaps, the size of all six faces is returned.
    unsigned width = getWidth(mipmap);
    unsigned height = getHeight(mipmap);
    unsigned depth = (getTextureType() == TEXTURE_CUBE)? 6 : getDepth(mipmap);

    if (channelFormat == COMPRESSED)
        return ((width + 3) >> 2) * ((height + 3) >> 2) * depth * getBytesPerBlock();

    return width * height * depth * getBytesPerPixel();
}

unsigned Image::getMipmapRangeSize(unsigned firstMipmap, unsigned mipmaps) const
{
    // Return the size of mipmap range (multiple consecutive mipmaps) in bytes.
    unsigned lastMipmap = firstMipmap + mipmaps;
    if (lastMipmap > mipmaps) lastMipmap = mipmaps;
    unsigned size = 0;
    for (unsigned i = firstMipmap; i < lastMipmap; i++)
        size += getMipmapSize(i);
    return size;
}

unsigned Image::getImageSize() const
{
    return getMipmapRangeSize(0, mipmaps);
}

Image::DataType Image::getDataType() const
{
    return dataType;
}

Image::ChannelFormat Image::getChannelFormat() const
{
    return channelFormat;
}

Image::TextureType Image::getTextureType() const
{
    if (depth == 0 && width == height)
        return TEXTURE_CUBE;
    else if(depth > 1)
        return TEXTURE_3D;
    else if (depth == 1)
        return TEXTURE_2D;
    else
        return TEXTURE_INVALID;
}

unsigned char *Image::getPixels()
{
    return data;
}

unsigned char *Image::getMipmapPixels(unsigned mipmap)
{
    if (getTextureType() == TEXTURE_CUBE)
        return 0;
    return data + getMipmapRangeSize(0, mipmap);
}

unsigned char *Image::getCubeMipmapPixels(unsigned cubeside, unsigned mipmap)
{
    if (getTextureType() != TEXTURE_CUBE)
        return 0;
    return data + (cubeside * getImageSize() + getMipmapRangeSize(0, mipmap)) / 6;
}

unsigned char *Image::get3DSliceMipmapPixels(unsigned slice, unsigned mipmap)
{
    if (getTextureType() != TEXTURE_3D)
        return 0;
    return getMipmapPixels(mipmap) + getWidth(mipmap) * getHeight(mipmap) * slice;
}

void Image::clear()
{
    if (data) free(data);
    data = NULL;
}

void Image::init(unsigned width, unsigned height, unsigned depth, unsigned mipmaps,
        DataType dataType, ChannelFormat channelFormat)
{
    clear();
    this->width = width;
    this->height = height;
    this->depth = depth;
    this->mipmaps = mipmaps;
    this->dataType = dataType;
    this->channelFormat = channelFormat;
#if 0
    printf("init: w=%i, h=%i, d=%i, mipmaps=%i, type=%s, channels=%s, textype=%s\n",
            width, height, depth, mipmaps, strFromDataType[dataType],
            strFromChannelFormat[channelFormat], strFromTextureType[getTextureType()]);
#endif
    data = (unsigned char*)malloc(getImageSize());
}

void Image::init(unsigned width, unsigned height, unsigned bpp)
{
    ChannelFormat channels;
    DataType type;

    // Guess the real format.
    switch (bpp)
    {
    case 8:
        type = UINT8;
        channels = L;
        break;
    case 16:
        type = UINT8;
        channels = LA;
        break;
    case 24:
        type = UINT8;
        channels = RGB;
        break;
    case 32:
        type = UINT8;
        channels = RGBA;
        break;
    case 48:
        type = UINT16;
        channels = RGB;
        break;
    case 64:
        type = UINT16;
        channels = RGBA;
        break;
    case 96:
        type = UINT32;
        channels = RGB;
        break;
    case 128:
        type = UINT32;
        channels = RGBA;
        break;
    default:
        std::cerr << "Image::init: Unsupported bpp: " << bpp << std::endl;
        return;
    }

    init(width, height, 1, 1, type, channels);
}

Image* Image::Create(std::string filename)
{
    std::string loader="default";
    std::string::size_type p = filename.rfind('.');
    if (p!=std::string::npos)
        loader = std::string(filename, p+1);
    return FactoryImage::CreateObject(loader, filename);
}

} // namespace io

} // namespace helper

} // namespace sofa

