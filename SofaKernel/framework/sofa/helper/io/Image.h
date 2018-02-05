/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_IO_IMAGE_H
#define SOFA_HELPER_IO_IMAGE_H

#include <cstdlib>
#include <sofa/helper/Factory.h>

#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace io
{
class SOFA_HELPER_API Image
{
public:
    enum DataType
    {
        UNORM8,  // unsigned char normalized to the range [0, 1] by the GPU
        UNORM16, // unsigned short normalized to the range [0, 1] by the GPU
        UINT32,  // unsigned int
        HALF,    // half float (10-bit mantissa, 5-bit exponent, 1-bit sign)
        FLOAT,   // ordinary float

        UCOMPRESSED, // compressed UNORM8
        // Compression schemes:
        // L,LA: luminance-alpha texture compression (as is defined by OpenGL)
        // R,RG: red-green texture compression (as is defined by OpenGL)
        // RGB:  DXT1 texture compression (S3TC in OpenGL)
        // RGBA: DXT5 texture compression (S3TC in OpenGL)
        // BGR:  non-applicable
        // BGRA: non-applicable

        COUNT_OF_DATA_TYPES
    };

    static const char *strFromDataType[COUNT_OF_DATA_TYPES+1];

    enum ChannelFormat
    {
        L,      // luminance
        LA,     // luminance, alpha
        R,      // red, green, blue, alpha...
        RG,
        RGB,
        RGBA,
        BGR,
        BGRA,

        COUNT_OF_CHANNEL_FORMATS
    };

    static const char *strFromChannelFormat[COUNT_OF_CHANNEL_FORMATS+1];

    enum TextureType
    {
        TEXTURE_2D,
        TEXTURE_3D,
        TEXTURE_CUBE,

        COUNT_OF_TEXTURE_TYPES,
        TEXTURE_INVALID = COUNT_OF_TEXTURE_TYPES
    };

    static const char *strFromTextureType[TEXTURE_INVALID+1];

    Image();
    virtual ~Image();
    Image(const Image& rhs);

    Image& operator=(const Image& rhs);



    unsigned getWidth(unsigned mipmap = 0) const;
    unsigned getHeight(unsigned mipmap = 0) const;
    unsigned getDepth(unsigned mipmap = 0) const;
    unsigned getBytesPerPixel() const;
    unsigned getBytesPerBlock() const;
    unsigned getBytesPerChannel() const;
    unsigned getChannelCount() const;
    unsigned getMipmapCount() const;
    unsigned getPixelCount() const;
    unsigned getLineSize(unsigned mipmap = 0) const;
    unsigned getMipmapSize(unsigned mipmap) const;
    unsigned getMipmapRangeSize(unsigned firstMipmap, unsigned mipmaps) const;
    unsigned getImageSize() const;
    DataType getDataType() const;
    ChannelFormat getChannelFormat() const;
    TextureType getTextureType() const;

    unsigned char *getPixels();
    unsigned char *getMipmapPixels(unsigned mipmap);
    unsigned char *getCubeMipmapPixels(unsigned cubeside, unsigned mipmap);
    unsigned char *get3DSliceMipmapPixels(unsigned slice, unsigned mipmap);

    void clear();

    // Please note that depth=0 denotes a cubemap, depth=1 denotes a 2D texture,
    // and depth>1 denotes a 3D texture.
    void init(unsigned width, unsigned height, unsigned depth, unsigned mipmaps,
            DataType dataType, ChannelFormat channelFormat);

    // for compatibility with the old interface
    void init(unsigned width, unsigned height, unsigned bpp);

    typedef Factory<std::string, Image, std::string> FactoryImage;

    static Image* Create(std::string filename);

    template<class Object>
    static Object* create(Object*, std::string arg = "")
    {
        return new Object(arg);
    }
	bool isLoaded() const { return (m_bLoaded>0); }

    virtual bool load(std::string filename);
    virtual bool save(std::string filename, int compression_level=-1);

protected:
	unsigned char m_bLoaded;

private:
    unsigned width, height, depth, mipmaps;
    DataType dataType;
    ChannelFormat channelFormat;
    unsigned char *data;
};

} // namespace io

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_HELPER_IO_IMAGE_CPP)
extern template class SOFA_HELPER_API Factory<std::string, io::Image, std::string>;
#endif

} // namespace helper

} // namespace sofa

/// This line register Image to the messaging system
MSG_REGISTER_CLASS(sofa::helper::io::Image, "Image")

#endif
