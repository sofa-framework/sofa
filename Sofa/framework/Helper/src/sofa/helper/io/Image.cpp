/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#define SOFA_HELPER_IO_IMAGE_CPP
#include <sofa/helper/io/Image.h>
#include <sofa/helper/Factory.inl>
#include <cstdio>
#include <cstring>


namespace sofa::helper
{

template class SOFA_HELPER_API Factory<std::string, sofa::helper::io::Image, std::string>;

namespace io
{

const char *Image::strFromDataType[COUNT_OF_DATA_TYPES+1] =
{
    "UNORM8",
    "UNORM16",
    "UINT32",
    "HALF",
    "FLOAT",

    "UCOMPRESSED",

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

    "COUNT_OF_CHANNEL_FORMATS"
};

const char *Image::strFromTextureType[TEXTURE_INVALID+1] =
{
    "TEXTURE_2D",
    "TEXTURE_3D",
    "TEXTURE_CUBE",

    "TEXTURE_INVALID"
};

static unsigned tableBytes[Image::COUNT_OF_DATA_TYPES][Image::COUNT_OF_CHANNEL_FORMATS] =
{
    // Bytes per pixel
    // UNORM8
    {
        1,  // L
        2,  // LA
        1,  // R
        2,  // RG
        3,  // RGB
        4,  // RGBA
        3,  // BGR
        4   // BGRA
    },
    // UNORM16
    {
        2,  // L
        4,  // LA
        2,  // R
        4,  // RG
        6,  // RGB
        8,  // RGBA
        6,  // BGR
        8   // BGRA
    },
    // UINT32
    {
        4,  // L
        8,  // LA
        4,  // R
        8,  // RG
        12, // RGB
        16, // RGBA
        12, // BGR
        16  // BGRA
    },
    // HALF
    {
        2,  // L
        4,  // LA
        2,  // R
        4,  // RG
        6,  // RGB
        8,  // RGBA
        6,  // BGR
        8   // BGRA
    },
    // FLOAT
    {
        4,  // L
        8,  // LA
        4,  // R
        8,  // RG
        12, // RGB
        16, // RGBA
        12, // BGR
        16  // BGRA
    },
    // Bytes per block
    // UCOMPRESSED
    {
        8,  // L
        16, // LA
        8,  // R
        16, // RG
        8,  // RGB
        16, // RGBA
        0,  // BGR
        0   // BGRA
    }
};

Image::Image()
    : m_bLoaded(0), m_data(nullptr)
{
}

Image::~Image()
{
    clear();
}

Image::Image(const Image& rhs)
    :m_data(nullptr)
{
    init(rhs.m_width,rhs.m_height,rhs.m_depth,rhs.m_mipmaps,rhs.m_dataType,rhs.m_channelFormat);
    memcpy(m_data,rhs.m_data,getImageSize());
    m_bLoaded = rhs.m_bLoaded;
}

Image& Image::operator=(const Image& rhs)
{
    if (this == &rhs)
        return *this;

    clear();
    init(rhs.m_width,rhs.m_height,rhs.m_depth,rhs.m_mipmaps,rhs.m_dataType,rhs.m_channelFormat);
    memcpy(m_data,rhs.m_data,getImageSize());
    m_bLoaded = rhs.m_bLoaded;
    return *this;
}

unsigned Image::getWidth(unsigned mipmap) const
{
    const unsigned result = m_width >> mipmap;
    return result ? result : 1;
}

unsigned Image::getHeight(unsigned mipmap) const
{
    const unsigned result = m_height >> mipmap;
    return result ? result : 1;
}

unsigned Image::getDepth(unsigned mipmap) const
{
    const unsigned result = m_depth >> mipmap;
    return result ? result : 1;
}

unsigned Image::getBytesPerPixel() const
{
    return m_dataType != UCOMPRESSED ? tableBytes[m_dataType][m_channelFormat] : 0;
}

unsigned Image::getBytesPerBlock() const
{
    return m_dataType == UCOMPRESSED ? tableBytes[UCOMPRESSED][m_channelFormat] : 0;
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
        2,  // LA
        1,  // R
        2,  // RG
        3,  // RGB
        4,  // RGBA
        3,  // BGR
        4   // BGRA
    };

    return table[m_channelFormat];
}

unsigned Image::getMipmapCount() const
{
    return m_mipmaps;
}

unsigned Image::getPixelCount() const
{
    const unsigned bytesPerPixel = getBytesPerPixel();
    return bytesPerPixel ? getImageSize() / getBytesPerPixel() : 0;
}

unsigned Image::getLineSize(unsigned mipmap) const
{
    return getWidth(mipmap) * getBytesPerPixel();
}

unsigned Image::getMipmapSize(unsigned mipmap) const
{
    // Return the size of one mipmap in bytes. For cubemaps, the size of all six faces is returned.
    const unsigned width = getWidth(mipmap);
    const unsigned height = getHeight(mipmap);
    const unsigned depth = (getTextureType() == TEXTURE_CUBE)? 6 : getDepth(mipmap);

    if (m_dataType == UCOMPRESSED)
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
    return getMipmapRangeSize(0, m_mipmaps);
}

Image::DataType Image::getDataType() const
{
    return m_dataType;
}

Image::ChannelFormat Image::getChannelFormat() const
{
    return m_channelFormat;
}

Image::TextureType Image::getTextureType() const
{
    if (m_dataType == UCOMPRESSED && m_channelFormat >= BGR)
        return TEXTURE_INVALID;

    if (m_depth == 0 && m_width == m_height)
        return TEXTURE_CUBE;
    else if(m_depth > 1)
        return TEXTURE_3D;
    else if (m_depth == 1)
        return TEXTURE_2D;
    else
        return TEXTURE_INVALID;
}

unsigned char *Image::getPixels()
{
    return m_data;
}

unsigned char *Image::getMipmapPixels(unsigned mipmap)
{
    if (getTextureType() == TEXTURE_CUBE)
        return nullptr;
    return m_data + getMipmapRangeSize(0, mipmap);
}

unsigned char *Image::getCubeMipmapPixels(unsigned cubeside, unsigned mipmap)
{
    if (getTextureType() != TEXTURE_CUBE)
        return nullptr;
    return m_data + (cubeside * getImageSize() + getMipmapRangeSize(0, mipmap)) / 6;
}

unsigned char *Image::get3DSliceMipmapPixels(unsigned slice, unsigned mipmap)
{
    if (getTextureType() != TEXTURE_3D)
        return nullptr;
    return getMipmapPixels(mipmap) + getWidth(mipmap) * getHeight(mipmap) * slice;
}

void Image::clear()
{
    if (m_data) free(m_data);
    m_data = nullptr;
    m_bLoaded = 0;
}

void Image::init(unsigned width, unsigned height, unsigned depth, unsigned mipmaps,
        DataType dataType, ChannelFormat channelFormat)
{
    clear();
    this->m_width = width;
    this->m_height = height;
    this->m_depth = depth;
    this->m_mipmaps = mipmaps;
    this->m_dataType = dataType;
    this->m_channelFormat = channelFormat;
#if 0
    printf("init: w=%i, h=%i, d=%i, mipmaps=%i, type=%s, channels=%s, textype=%s\n",
            width, height, depth, mipmaps, strFromDataType[dataType],
            strFromChannelFormat[channelFormat], strFromTextureType[getTextureType()]);
#endif
    m_data = (unsigned char*)malloc(getImageSize());
}

void Image::init(unsigned width, unsigned height, unsigned bpp)
{
    ChannelFormat channels;
    DataType type;

    // Guess the real format.
    switch (bpp)
    {
    case 8:
        type = UNORM8;
        channels = L;
        break;
    case 16:
        type = UNORM8;
        channels = LA;
        break;
    case 24:
        type = UNORM8;
        channels = RGB;
        break;
    case 32:
        type = UNORM8;
        channels = RGBA;
        break;
    case 48:
        type = UNORM16;
        channels = RGB;
        break;
    case 64:
        type = UNORM16;
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
        msg_error() << "init(): Unsupported bpp: " << bpp << msgendl;
        return;
    }

    init(width, height, 1, 1, type, channels);
}


bool Image::load(std::string filename)
{
    SOFA_UNUSED(filename);

    msg_warning() << "This Image format did not implement load()";
    return false;
}

bool Image::save(std::string filename, int compression_level)
{
    SOFA_UNUSED(filename);
    SOFA_UNUSED(compression_level);

    msg_warning() << "This Image format did not implement save()";
    return false;
}

Image* Image::Create(std::string filename)
{
    std::string extension="default";
    const std::string::size_type p = filename.rfind('.');
    if (p!=std::string::npos)
        extension = std::string(filename, p+1);
    Image* createdImage = FactoryImage::CreateObject(extension, filename);
    if( extension != "default" )
    {
        if(!createdImage )
        {
            type::vector<std::string> validExtensions;
            helper::io::Image::FactoryImage::getInstance()->uniqueKeys(std::back_inserter(validExtensions));
            msg_error("Image") << "Could not load image with extension " << extension << ". Valid extensions: " << validExtensions;
        }
    }
    else
    {
        type::vector<std::string> validExtensions;
        helper::io::Image::FactoryImage::getInstance()->uniqueKeys(std::back_inserter(validExtensions));
        msg_error("Image") << "No extension detected. Valid extensions: " << validExtensions;
    }
    return createdImage;
}

} // namespace io

} // namespace sofa::helper



