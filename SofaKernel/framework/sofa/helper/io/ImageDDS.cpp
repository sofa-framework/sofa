/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/io/ImageDDS.h>
#include <sofa/helper/system/FileRepository.h>
#include <iostream>
#ifdef PS3
#include <stdio.h>
#else
#include <cstdio>		// fopen and friends
#endif

#define _FOURCC(a, b, c, d) (a | (b << 8) | (c << 16) | (d << 24))

// Basic DDS definitions

#define DDS_MAGIC        _FOURCC('D', 'D', 'S', ' ')

#define DDPF_ALPHAPIXELS 0x00000001
#define DDPF_FOURCC      0x00000004
#define DDPF_RGB         0x00000040
#define DDPF_LUMINANCE   0x00020000

#define DDSD_CAPS        0x00000001
#define DDSD_HEIGHT      0x00000002
#define DDSD_WIDTH       0x00000004
#define DDSD_PITCH       0x00000008
#define DDSD_PIXELFORMAT 0x00001000
#define DDSD_MIPMAPCOUNT 0x00020000
#define DDSD_LINEARSIZE  0x00080000
#define DDSD_DEPTH       0x00800000

#define DDSCAPS_COMPLEX  0x00000008
#define DDSCAPS_TEXTURE  0x00001000
#define DDSCAPS_MIPMAP   0x00400000

#define DDSCAPS2_VOLUME  0x00200000
#define DDSCAPS2_CUBEMAP 0x00000200

#define DDSCAPS2_CUBEMAP_POSITIVEX 0x00000400
#define DDSCAPS2_CUBEMAP_NEGATIVEX 0x00000800
#define DDSCAPS2_CUBEMAP_POSITIVEY 0x00001000
#define DDSCAPS2_CUBEMAP_NEGATIVEY 0x00002000
#define DDSCAPS2_CUBEMAP_POSITIVEZ 0x00004000
#define DDSCAPS2_CUBEMAP_NEGATIVEZ 0x00008000

#define DDSCAPS2_CUBEMAP_ALL_FACES (DDSCAPS2_CUBEMAP_POSITIVEX | DDSCAPS2_CUBEMAP_NEGATIVEX | DDSCAPS2_CUBEMAP_POSITIVEY | \
                                    DDSCAPS2_CUBEMAP_NEGATIVEY | DDSCAPS2_CUBEMAP_POSITIVEZ | DDSCAPS2_CUBEMAP_NEGATIVEZ)

// DDS non-standard file formats

#define DDS_FORMAT_DXT1     _FOURCC('D', 'X', 'T', '1')
#define DDS_FORMAT_DXT3     _FOURCC('D', 'X', 'T', '3')
#define DDS_FORMAT_DXT5     _FOURCC('D', 'X', 'T', '5')
#define DDS_FORMAT_ATI1N    _FOURCC('A', 'T', 'I', '1')
#define DDS_FORMAT_ATI2N    _FOURCC('A', 'T', 'I', '2')

#define DDS_FORMAT_RG16     34 // ATI-specific, maybe?
#define DDS_FORMAT_RGBA16   36
#define DDS_FORMAT_R16F     111
#define DDS_FORMAT_RG16F    112
#define DDS_FORMAT_RGBA16F  113
#define DDS_FORMAT_R32F     114
#define DDS_FORMAT_RG32F    115
#define DDS_FORMAT_RGBA32F  116
#define DDS_RBIT_A2_RGB10   0x3FF00000
#define DDS_GBIT_A2_RGB10   0x000FFC00
#define DDS_BBIT_A2_RGB10   0x000003FF
#define DDS_ABIT_A2_RGB10   0xC0000000

#define DDS_RBIT_RG3_B2     0xE0
#define DDS_GBIT_RG3_B2     0x1C
#define DDS_BBIT_RG3_B2     0x03

namespace sofa
{
namespace helper
{
namespace io
{
SOFA_DECL_CLASS(ImageDDS)

Creator<Image::FactoryImage,ImageDDS> ImageDDSClass("dds");

#pragma pack (push, 1)

struct DDSHeader
{
    uint32_t dwMagic;
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwHeight;
    uint32_t dwWidth;
    uint32_t dwPitchOrLinearSize;
    uint32_t dwDepth;
    uint32_t dwMipMapCount;
    uint32_t dwReserved[11];

    struct DDSPixelFormat
    {
        uint32_t dwSize;
        uint32_t dwFlags;
        uint32_t dwFourCC;
        uint32_t dwRGBBitCount;
        uint32_t dwRBitMask;
        uint32_t dwGBitMask;
        uint32_t dwBBitMask;
        uint32_t dwRGBAlphaBitMask;
    } ddpfPixelFormat;

    struct DDSCaps
    {
        uint32_t dwCaps1;
        uint32_t dwCaps2;
        uint32_t Reserved[2];
    } ddsCaps;

    uint32_t dwReserved2;
};

#pragma pack (pop)

bool ImageDDS::load(const std::string &filename)
{
    FILE *file = fopen(filename.c_str(), "rb");
    m_bLoaded = 0;

    if (!file)
    {
        msg_error("ImageDDS") << "load: Cannot open file '" << filename << "'";
        return false;
    }

    // Load the file header
    DDSHeader header;
    if (!fread(&header, sizeof(header), 1, file))
    {
        fclose(file);
        msg_error("ImageDDS") << "load: Cannot read the DDS header from '" << filename << "'";
        return false;
    }

    // Check the first 4 bytes
    if (header.dwMagic != DDS_MAGIC)
    {
        fclose(file);
        msg_error("ImageDDS") << "load: File '" << filename << "' is not of the DDS format.";
        return false;
    }

    Image::DataType type;
    Image::ChannelFormat channels;

    // Parse the format
    switch (header.ddpfPixelFormat.dwFourCC)
    {
        // FourCC formats
    case DDS_FORMAT_DXT1:    type = Image::UCOMPRESSED; channels = Image::RGB; break;
        //case DDS_FORMAT_DXT3:    type = Image::UCOMPRESSED; channels = Image::RGBA; break;
    case DDS_FORMAT_DXT5:    type = Image::UCOMPRESSED; channels = Image::RGBA; break;
    case DDS_FORMAT_ATI1N:   type = Image::UCOMPRESSED; channels = Image::L; break;
    case DDS_FORMAT_ATI2N:   type = Image::UCOMPRESSED; channels = Image::LA; break;

    case DDS_FORMAT_RG16:    type = Image::UNORM16; channels = Image::LA;         break;
        //case DDS_FORMAT_RGB16:   type = Image::UNORM16; channels = Image::RGB;        break;
    case DDS_FORMAT_RGBA16:  type = Image::UNORM16; channels = Image::RGBA;       break;
        //case DDS_FORMAT_BGR16:   type = Image::UNORM16; channels = Image::BGR;        break;
        //case DDS_FORMAT_BGRA16:  type = Image::UNORM16; channels = Image::BGRA;       break;

        /*case DDS_FORMAT_L32:     type = Image::UINT32; channels = Image::L;          break;
        case DDS_FORMAT_LA32:    type = Image::UINT32; channels = Image::LA;         break;
        case DDS_FORMAT_R32:     type = Image::UINT32; channels = Image::R;          break;
        case DDS_FORMAT_RG32:    type = Image::UINT32; channels = Image::RG;         break;
        case DDS_FORMAT_RGB32:   type = Image::UINT32; channels = Image::RGB;        break;
        case DDS_FORMAT_RGBA32:  type = Image::UINT32; channels = Image::RGBA;       break;
        case DDS_FORMAT_BGR32:   type = Image::UINT32; channels = Image::BGR;        break;
        case DDS_FORMAT_BGRA32:  type = Image::UINT32; channels = Image::BGRA;       break;*/

        //case DDS_FORMAT_L16F:    type = Image::HALF;   channels = Image::L;          break;
        //case DDS_FORMAT_LA16F:   type = Image::HALF;   channels = Image::LA;         break;
    case DDS_FORMAT_R16F:    type = Image::HALF;   channels = Image::R;          break;
    case DDS_FORMAT_RG16F:   type = Image::HALF;   channels = Image::RG;         break;
        //case DDS_FORMAT_RGB16F:  type = Image::HALF;   channels = Image::RGB;        break;
    case DDS_FORMAT_RGBA16F: type = Image::HALF;   channels = Image::RGBA;       break;
        //case DDS_FORMAT_BGR16F:  type = Image::HALF;   channels = Image::BGR;        break;
        //case DDS_FORMAT_BGRA16F: type = Image::HALF;   channels = Image::BGRA;       break;

        //case DDS_FORMAT_L32F:    type = Image::FLOAT;  channels = Image::L;          break;
        //case DDS_FORMAT_LA32F:   type = Image::FLOAT;  channels = Image::LA;         break;
    case DDS_FORMAT_R32F:    type = Image::FLOAT;  channels = Image::R;          break;
    case DDS_FORMAT_RG32F:   type = Image::FLOAT;  channels = Image::RG;         break;
        //case DDS_FORMAT_RGB32F:  type = Image::FLOAT;  channels = Image::RGB;        break;
    case DDS_FORMAT_RGBA32F: type = Image::FLOAT;  channels = Image::RGBA;       break;
        //case DDS_FORMAT_BGR32F:  type = Image::FLOAT;  channels = Image::BGR;        break;
        //case DDS_FORMAT_BGRA32F: type = Image::FLOAT;  channels = Image::BGRA;       break;

    default:
    {
        bool error = false;

        switch (header.ddpfPixelFormat.dwRGBBitCount)
        {
            // 8bit formats (R8, L8)
        case 8:
            if      (header.ddpfPixelFormat.dwRBitMask == 0xFF &&
                    header.ddpfPixelFormat.dwGBitMask == 0x00 &&
                    header.ddpfPixelFormat.dwBBitMask == 0x00)
            {
                type = Image::UNORM8;
                channels = Image::R;
            }
            else if (header.ddpfPixelFormat.dwRBitMask == DDS_RBIT_RG3_B2)
                error = true;   // RG3_B2 is not supported
            else
            {
                type = Image::UNORM8;
                channels = Image::L;
            }
            break;

            // 16bit formats (AL8, GR8, R16, L16)
        case 16:
            if      (header.ddpfPixelFormat.dwRGBAlphaBitMask)
            {
                type = Image::UNORM8;
                channels = Image::LA;
            }
            else if (header.ddpfPixelFormat.dwRBitMask == 0x00FF &&
                    header.ddpfPixelFormat.dwGBitMask == 0xFF00 &&
                    header.ddpfPixelFormat.dwBBitMask == 0x0000)
            {
                type = Image::UNORM8;
                channels = Image::RG;
            }
            else if (header.ddpfPixelFormat.dwRBitMask == 0xFFFF &&
                    header.ddpfPixelFormat.dwGBitMask == 0x0000 &&
                    header.ddpfPixelFormat.dwBBitMask == 0x0000)
            {
                type = Image::UNORM16;
                channels = Image::R;
            }
            else
            {
                type = Image::UNORM16;
                channels = Image::L;
            }
            break;

            // 24bit formats (RGB8, BGR8)
        case 24:
            if (header.ddpfPixelFormat.dwRBitMask == 0x0000FF &&
                header.ddpfPixelFormat.dwGBitMask == 0x00FF00 &&
                header.ddpfPixelFormat.dwGBitMask == 0xFF0000)
            {
                type = Image::UNORM8;
                channels = Image::RGB;
            }
            else
            {
                type = Image::UNORM8;
                channels = Image::BGR;
            }
            break;

            // 32bit formats (ARGB8, ABGR8)
        case 32:
            /*if (header.ddpfPixelFormat.dwRBitMask == DDS_RBIT_A2_RGB10)
            {
                type = Image::UINT32;
                channels = Image::A2_RGB10;
            }
            else*/ if (header.ddpfPixelFormat.dwRBitMask == 0x000000FF &&
                    header.ddpfPixelFormat.dwGBitMask == 0x0000FF00 &&
                    header.ddpfPixelFormat.dwBBitMask == 0x00FF0000)
            {
                type = Image::UNORM8;
                channels = Image::RGBA;
            }
            else
            {
                type = Image::UNORM8;
                channels = Image::BGRA;
            }
            break;

        default:
            error = true;
        }

        // The loader doesn't support this format.
        if (error)
        {
            fclose(file);
            msg_error("ImageDDS") << "load: File '" << filename << "' has unknown or unsupported format.";
            return false;
        }
    }
    }

    int depth = (header.ddsCaps.dwCaps2 & DDSCAPS2_CUBEMAP)? 0 :
            (!header.dwDepth)? 1 : header.dwDepth;
    int mipmaps = (header.dwMipMapCount <= 0)? 1 : header.dwMipMapCount;

    // Load the content of the file.
    init(header.dwWidth, header.dwHeight, depth, mipmaps, type, channels);
    std::size_t size = getImageSize();
    std::size_t read = fread(getPixels(), 1, size, file);
    fclose(file);
    if (read != size)
    {
        msg_error("ImageDDS") << "load: Cannot read file '" + filename + "', a part of the file is missing.";
        return false; // "
    }

    msg_info("ImageDDS") << "DDS image " << filename << ": Type: " << strFromTextureType[getTextureType()]
                         << ", Size: " << getWidth() << "x" << getHeight() << "x" << getDepth()
                         << ", Format: " << strFromDataType[getDataType()] << ", Channels: " << strFromChannelFormat[getChannelFormat()]
                         << ", Mipmaps: " << getMipmapCount() ;

    m_bLoaded = 1;
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static DDSHeader::DDSPixelFormat pixelFormatTable[Image::COUNT_OF_DATA_TYPES][Image::COUNT_OF_CHANNEL_FORMATS] =
{
    // UNORM8
    {
        {32, DDPF_LUMINANCE, 0, 8, 0xFF, 0xFF, 0xFF, 0},                                // L
        {32, DDPF_LUMINANCE | DDPF_ALPHAPIXELS, 0, 16, 0x00FF, 0x00FF, 0x00FF, 0xFF00}, // LA
        {32, DDPF_RGB, 0, 8, 0xFF, 0, 0, 0},                                            // R
        {32, DDPF_RGB, 0, 16, 0x00FF, 0xFF00, 0, 0},                                    // RG
        {32, DDPF_RGB, 0, 24, 0x0000FF, 0x00FF00, 0xFF0000, 0},                         // RGB
        {32, DDPF_RGB | DDPF_ALPHAPIXELS, 0, 32, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000},   // RGBA
        {32, DDPF_RGB, 0, 24, 0xFF0000, 0x00FF00, 0x0000FF, 0},                         // BGR
        {32, DDPF_RGB | DDPF_ALPHAPIXELS, 0, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000}    // BGRA
    },
    // UNORM16
    {
        {32, DDPF_LUMINANCE, 0, 16, 0xFFFF, 0xFFFF, 0xFFFF, 0}, // L
        {32, DDPF_FOURCC, DDS_FORMAT_RG16, 0, 0, 0, 0, 0},      // LA - basically the same as RG
        {32, DDPF_RGB, 0, 16, 0xFFFF, 0, 0, 0},                 // R
        {32, DDPF_FOURCC, DDS_FORMAT_RG16, 0, 0, 0, 0, 0},      // RG
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_RGB16, 0, 0, 0, 0, 0},     // RGB
        {32, DDPF_FOURCC, DDS_FORMAT_RGBA16, 0, 0, 0, 0, 0},    // RGBA
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_BGR16, 0, 0, 0, 0, 0},     // BGR
        {0, 0, 0, 0, 0, 0, 0, 0}  // {32, DDPF_FOURCC, DDS_FORMAT_BGRA16, 0, 0, 0, 0, 0}     // BGRA
    },
    // UINT32
    {
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_L32, 0, 0, 0, 0, 0},       // L
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_LA32, 0, 0, 0, 0, 0},      // LA
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_R32, 0, 0, 0, 0, 0},       // R
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_RG32, 0, 0, 0, 0, 0},      // RG
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_RGB32, 0, 0, 0, 0, 0},     // RGB
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_RGBA32, 0, 0, 0, 0, 0},    // RGBA
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_BGR32, 0, 0, 0, 0, 0},     // BGR
        {0, 0, 0, 0, 0, 0, 0, 0}  // {32, DDPF_FOURCC, DDS_FORMAT_BGRA32, 0, 0, 0, 0, 0}     // BGRA
    },
    // HALF
    {
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_L16F, 0, 0, 0, 0, 0},      // L
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_LA16F, 0, 0, 0, 0, 0},     // LA
        {32, DDPF_FOURCC, DDS_FORMAT_R16F, 0, 0, 0, 0, 0},      // R
        {32, DDPF_FOURCC, DDS_FORMAT_RG16F, 0, 0, 0, 0, 0},     // RG
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_RGB16F, 0, 0, 0, 0, 0},    // RGB
        {32, DDPF_FOURCC, DDS_FORMAT_RGBA16F, 0, 0, 0, 0, 0},   // RGBA
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_BGR16F, 0, 0, 0, 0, 0},    // BGR
        {0, 0, 0, 0, 0, 0, 0, 0}  // {32, DDPF_FOURCC, DDS_FORMAT_BGRA16F, 0, 0, 0, 0, 0}    // BGRA
    },
    // FLOAT
    {
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_L32F, 0, 0, 0, 0, 0},      // L
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_LA32F, 0, 0, 0, 0, 0},     // LA
        {32, DDPF_FOURCC, DDS_FORMAT_R32F, 0, 0, 0, 0, 0},      // R
        {32, DDPF_FOURCC, DDS_FORMAT_RG32F, 0, 0, 0, 0, 0},     // RG
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_RGB32F, 0, 0, 0, 0, 0},    // RGB
        {32, DDPF_FOURCC, DDS_FORMAT_RGBA32F, 0, 0, 0, 0, 0},   // RGBA
        {0, 0, 0, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_BGR32F, 0, 0, 0, 0, 0},    // BGR
        {0, 0, 0, 0, 0, 0, 0, 0}  // {32, DDPF_FOURCC, DDS_FORMAT_BGRA32F, 0, 0, 0, 0, 0}    // BGRA
    },
    // UCOMPRESSED
    {
        {32, DDPF_FOURCC, DDS_FORMAT_ATI1N, 0, 0, 0, 0, 0},
        {32, DDPF_FOURCC, DDS_FORMAT_ATI2N, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {32, DDPF_FOURCC, DDS_FORMAT_DXT1, 0, 0, 0, 0, 0},
        {32, DDPF_FOURCC, DDS_FORMAT_DXT5, 0, 0, 0, 0, 0}, // {32, DDPF_FOURCC, DDS_FORMAT_DXT3, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    },
};

bool ImageDDS::save(const std::string &filename, int)
{
    // Fill in the header
    DDSHeader header;
    header.dwMagic = DDS_MAGIC;
    header.dwSize = 124;
    header.dwFlags = DDSD_CAPS | DDSD_PIXELFORMAT | DDSD_WIDTH | DDSD_HEIGHT;
    header.dwFlags |= (getDataType() != Image::UCOMPRESSED)? DDSD_PITCH : DDSD_LINEARSIZE;
    header.dwFlags |= (getMipmapCount() > 1)? DDSD_MIPMAPCOUNT : 0;
    header.dwFlags |= (getTextureType() == Image::TEXTURE_3D)? DDSD_DEPTH : 0;
    header.dwHeight = getHeight(0);
    header.dwWidth = getWidth(0);
    header.dwPitchOrLinearSize = (getDataType() != Image::UCOMPRESSED)?
            header.dwWidth * getBytesPerPixel() :
            getMipmapSize(0) / ((getTextureType() == Image::TEXTURE_CUBE)? 6 : 1);
    header.dwDepth = (getTextureType() == Image::TEXTURE_3D)? getDepth(0) : 0;
    header.dwMipMapCount = (getMipmapCount() > 1)? getMipmapCount() : 0;
    std::fill_n(header.dwReserved, 11, 0);

    header.ddpfPixelFormat = pixelFormatTable[getDataType()][getChannelFormat()];
    if (!header.ddpfPixelFormat.dwSize)
    {
        msg_error("ImageDDS") << "save: Cannot save file '" << filename << "', the image format is unsupported." ;
        return false;
    }

    header.ddsCaps.dwCaps1 = DDSCAPS_TEXTURE;
    header.ddsCaps.dwCaps1 |= (getMipmapCount() > 1)? DDSCAPS_MIPMAP : 0;
    header.ddsCaps.dwCaps1 |= (getMipmapCount() > 1 || getTextureType() == Image::TEXTURE_3D ||
            getTextureType() == Image::TEXTURE_CUBE)? DDSCAPS_COMPLEX : 0;
    header.ddsCaps.dwCaps2 = 0;
    header.ddsCaps.dwCaps2 |= (getTextureType() == Image::TEXTURE_3D)? DDSCAPS2_VOLUME : 0;
    header.ddsCaps.dwCaps2 |= (getTextureType() == Image::TEXTURE_CUBE)? DDSCAPS2_CUBEMAP | DDSCAPS2_CUBEMAP_ALL_FACES : 0;
    std::fill_n(header.ddsCaps.Reserved, 2, 0);
    header.dwReserved2 = 0;

    // Save the file
    FILE *file = fopen(filename.c_str(), "wb");
    if (!file)
    {
        msg_error("ImageDDS") << "save: Cannot open file '" << filename << "' for writing." ;
        return false;
    }

    bool isWriteOk = true;
    isWriteOk = isWriteOk && fwrite(&header, sizeof(DDSHeader), 1, file) == sizeof(DDSHeader);
    isWriteOk = isWriteOk && fwrite(getPixels(), getImageSize(), 1, file) == getImageSize();
    fclose(file);
    if (!isWriteOk)
    {
        msg_error("ImageDDS") << "save: Cannot write to file '" << filename << "'";
    }
    return isWriteOk;
}
} // namespace io
} // namespace helper
} // namespace sofa
