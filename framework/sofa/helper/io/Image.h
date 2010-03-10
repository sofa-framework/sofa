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
#ifndef SOFA_HELPER_IO_IMAGE_H
#define SOFA_HELPER_IO_IMAGE_H

#include <stdlib.h>
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
        UINT8,
        UINT16,
        UINT32,
        HALF,
        FLOAT,

        COMPRESSED_DXT1,
        COMPRESSED_DXT3,
        COMPRESSED_DXT5,
        COMPRESSED_L,
        COMPRESSED_LA,

        COUNT_OF_DATA_TYPES
    };

    static const char *strFromDataType[COUNT_OF_DATA_TYPES+1];

    enum ChannelFormat
    {
        L,
        LA,
        R,
        RG,
        RGB,
        RGBA,
        BGR,
        BGRA,
        COMPRESSED,

        COUNT_OF_CHANNEL_FORMATS
    };

    static const char *strFromChannelFormat[COUNT_OF_CHANNEL_FORMATS+1];

    enum TextureType
    {
        TEXTURE_2D,
        TEXTURE_3D,
        TEXTURE_CUBE,

        TEXTURE_INVALID
    };

    static const char *strFromTextureType[TEXTURE_INVALID+1];

    Image();
    virtual ~Image();

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
    void init(unsigned width, unsigned height, unsigned depth, unsigned mipmaps,
            DataType dataType, ChannelFormat channelFormat);

    // for compatibility with the old interface
    void init(unsigned width, unsigned height, unsigned bpp);

    typedef Factory<std::string, Image, std::string> FactoryImage;

    static Image* Create(std::string filename);

    template<class Object>
    static void create(Object*& obj, std::string arg)
    {
        obj = new Object(arg);
    }

private:
    unsigned width, height, depth, mipmaps;
    DataType dataType;
    ChannelFormat channelFormat;
    unsigned char *data;
};

} // namespace io

#if defined(WIN32) && !defined(SOFA_BUILD_HELPER)
extern template class SOFA_HELPER_API Factory<std::string, io::Image, std::string>;
#endif

} // namespace helper

} // namespace sofa

#endif
