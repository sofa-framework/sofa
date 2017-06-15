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
#include <sofa/helper/io/ImageBMP.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/logging/Messaging.h>
#include <iostream>
#ifdef PS3
#include <stdio.h>
#else
#include <cstdio>		// fopen and friends
#endif

namespace sofa
{

namespace helper
{

namespace io
{

//using namespace sofa::defaulttype;

SOFA_DECL_CLASS(ImageBMP)

Creator<Image::FactoryImage,ImageBMP> ImageBMPClass("bmp");

bool ImageBMP::load(std::string filename)
{
    m_bLoaded = 0;

    if (!sofa::helper::system::DataRepository.findFile(filename))
    {
        msg_error("ImageBMP") << "File " << filename << " not found ";
        return false;
    }
    unsigned short int bfType;
    uint32_t bfOffBits;
    short int biPlanes;
    short int biBitCount;
    long int biSizeImage;
    FILE *file;
    /* make sure the file is there and open it read-only (binary) */
    if ((file = fopen(filename.c_str(), "rb")) == NULL)
    {
        msg_error("ImageBMP") << "File not found : " << filename;
        return false;
    }

    if(!fread(&bfType, sizeof(short int), 1, file))
    {
        msg_error("ImageBMP") << "Unable to read file!";
        fclose(file);
        return false;
    }

    /* check if file is a bitmap */
    if (bfType != 19778)
    {
        msg_error("ImageBMP") << "Not a Bitmap-File!";
        fclose(file);
        return false;
    }
    /* get the file size */
    /* skip file size and reserved fields of bitmap file header */
    fseek(file, 8, SEEK_CUR);
    /* get the position of the actual bitmap data */
    if (!fread(&bfOffBits, sizeof(uint32_t), 1, file))
    {
        msg_error("ImageBMP") << "Unable to read file!";
        fclose(file);
        return false;
    }
    // printf("Data at Offset: %ld\n", bfOffBits);
    /* skip size of bitmap info header */
    fseek(file, 4, SEEK_CUR);
    /* get the width of the bitmap */
    int width;
    if (fread(&width, sizeof(int), 1, file) != 1)
    {
        msg_error("ImageBMP") << "Error: fread can't read the width of the bitmap.";
        fclose(file);
        return false;
    }

    if (width < 0) width = -width;
    //printf("Width of Bitmap: %d\n", texture->width);
    /* get the height of the bitmap */
    int height;
    if (fread(&height, sizeof(int), 1, file) != 1)
    {
        msg_error("ImageBMP") << "Error: fread can't read the height of the bitmap.";
        fclose(file);
        return false;
    }

    bool upsidedown = false;
    if (height < 0) { height = -height; upsidedown = true; }
    //printf("Height of Bitmap: %d\n", texture->height);
    /* get the number of planes (must be set to 1) */
    if (fread(&biPlanes, sizeof(short int), 1, file) != 1)
    {
        msg_error("ImageBMP") << "Error: fread can't read the number of planes.";
        fclose(file);
        return false;
    }

    if (biPlanes != 1)
    {
        msg_error("ImageBMP") << "Error: number of Planes not 1!";
        fclose(file);
        return false;
    }
    /* get the number of bits per pixel */
    if (!fread(&biBitCount, sizeof(short int), 1, file))
    {
        msg_error("ImageBMP") << "Error reading file!";
        fclose(file);
        return false;
    }
    //printf("Bits per Pixel: %d\n", biBitCount);
    if (biBitCount != 24 && biBitCount != 32 && biBitCount != 8)
    {
        msg_error("ImageBMP") << "Bits per Pixel not supported";
        fclose(file);
        return false;
    }
    int nbBits = biBitCount;
    int nc = ((nbBits+7)/8);
    /* calculate the size of the image in bytes */
    biSizeImage = width * height * nc;
#ifndef NDEBUG
    //std::cout << "Size of the image data: " << biSizeImage << std::endl;
    //std::cout << "ImageBMP "<<filename<<" "<<width<<"x"<<height<<"x"<<nbBits<<" = "<<biSizeImage<<" bytes"<<std::endl;
#endif

    Image::ChannelFormat channels;
    switch (nc)
    {
    case 4:
        channels = Image::RGBA;
        break;
    case 3:
        channels = Image::RGB;
        break;
    case 1:
        channels = Image::L;
        break;
    default:
        fprintf(stderr, "ImageBMP: Unsupported number of bits per pixel: %i\n", nc*8);
        fclose(file);
        return false;
    }
    init(width, height, 1, 1, Image::UNORM8, channels);
    unsigned char *data = getPixels();
    /* seek to the actual data */
    fseek(file, bfOffBits, SEEK_SET);

    if (((width*nc)%4)==0 && !upsidedown)
    {
        if (!fread(data, biSizeImage, 1, file))
        {
            msg_error("ImageBMP") << "Unable to load file!";
            fclose(file);
            return false;
        }
    }
    else
    {
        int pad = (4-((width*nc)%4))%4;
        char buf[3];
        for (int y=0; y<height; y++)
        {
            if (!fread(data+(upsidedown?height-1-y:y)*width*nc, width*nc, 1, file))
            {
                msg_error("ImageBMP") << "Unable to load file!";
                fclose(file);
                return false;
            }
            if (pad && !fread(buf, 4-((width*nc)%4), 1, file))
            {
                msg_error("ImageBMP") << "Unable to load file!";
                fclose(file);
                return false;
            }
        }
    }

    if (nc == 3 || nc == 4)
    {
        int i;
        // swap red and blue (bgr -> rgb)
        for (i = 0; i < width*height*nc; i += nc)
        {
            unsigned char temp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = temp;
        }
    }

    fclose(file);
    m_bLoaded = 1;
    return true;
}

static bool fwriteW(FILE* file, unsigned short data)
{
    return fwrite(&data,sizeof(data),1,file)!=0;
}

static bool fwriteDW(FILE* file, unsigned long data)
{
    return fwrite(&data,sizeof(data),1,file)!=0;
}

bool ImageBMP::save(std::string filename, int)
{
    FILE *file;
    /* make sure the file is there and open it read-only (binary) */
    if ((file = fopen(filename.c_str(), "wb")) == NULL)
    {
        msg_error("ImageBMP") << "File write access failed : " << filename ;
        return false;
    }

    unsigned long lineSizeIn = getLineSize(0);
    unsigned long lineSizeOut = ((lineSizeIn+3)/4)*4;
    unsigned long biSizeImage = lineSizeOut*getHeight(0);

    // BITMAPFILEHEADER
    if (!fwriteW(file, (unsigned short)'B' | ((unsigned short)'M' << 8))) return false; // Type
    if (!fwriteDW(file, 14+40+biSizeImage)) return false; // Size
    if (!fwriteW(file, 0)) return false; // Reserved1
    if (!fwriteW(file, 0)) return false; // Reserved2
    if (!fwriteDW(file, 14+40)) return false; // OffBits

    // BITMAPINFOHEADER
    const unsigned width = getWidth(0);
    const unsigned height = getHeight(0);
    const unsigned int bytespp = getBytesPerPixel();
    if (!fwriteDW(file, 40)) return false; // Size
    if (!fwriteDW(file, width)) return false; // Width
    if (!fwriteDW(file, height)) return false; // Height
    if (!fwriteW(file, 1)) return false; // Planes
    if (!fwriteW(file, bytespp*8)) return false; // BitCount
    if (!fwriteDW(file, 0)) return false; // Compression
    if (!fwriteDW(file, biSizeImage)) return false; // SizeImage
    if (!fwriteDW(file, 2825)) return false; // XPelsPerMeter
    if (!fwriteDW(file, 2825)) return false; // YPelsPerMeter
    if (!fwriteDW(file, 0)) return false; // ClrUsed
    if (!fwriteDW(file, 0)) return false; // ClrImportant

    unsigned char *data = getPixels();
    if(bytespp==3)
    {
        /* swap red and blue (rgb -> bgr) */
        for (unsigned i = 0; i < width*height*3; i += 3)
        {
            unsigned char temp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = temp;
        }
    }
    if (lineSizeIn == lineSizeOut)
    {
        if (!fwrite(data, biSizeImage, 1, file)) return false;
    }
    else
    {
        for (unsigned y=0; y<height; y++)
        {
            if (!fwrite(data+y*lineSizeIn, lineSizeIn, 1, file)) return false;
            char buf[3]= {0,0,0};
            if (!fwrite(buf, lineSizeOut-lineSizeIn, 1, file)) return false;
        }
    }
    if(bytespp==3)
    {
        unsigned i;
        /* swap red and blue (bgr -> rgb) */
        for (i = 0; i < width*height*3; i += 3)
        {
            unsigned char temp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = temp;
        }
    }

    fclose(file);
    return true;
}

} // namespace io

} // namespace helper

} // namespace sofa

