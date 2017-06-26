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
#include <sofa/helper/io/ImagePNG.h>
#include <sofa/helper/system/FileRepository.h>
#include <iostream>

#ifdef SOFA_HAVE_PNG
#include <png.h>
#ifdef _MSC_VER
#ifdef _DEBUG
#pragma comment(lib,"libpngd.lib")
#pragma comment(lib,"zlibd.lib")
#else
#pragma comment(lib,"libpng.lib")
#pragma comment(lib,"zlib.lib")
#endif
#endif
#endif

#if PNG_LIBPNG_VER >= 10209
#define RECENT_LIBPNG
#endif

namespace sofa
{

namespace helper
{

namespace io
{

//using namespace sofa::defaulttype;

SOFA_DECL_CLASS(ImagePNG)

#ifdef SOFA_HAVE_PNG

// Set the compression level. The valid values for "COMPRESSION_LEVEL" range from [0,9]
// The value 0 implies no compression and 9 implies maximal compression
// The value -1 implies default compression (level 6)

Creator<Image::FactoryImage,ImagePNG> ImagePNGClass("png");

// we have to give our own reading function to libpng in order to use the "fread" function belonging to Sofa
// and not the one belonging to libpng since Sofa do the "fopen" on the FILE struct
void png_my_read_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
   png_size_t check;

   if (png_ptr == NULL)
      return;

   /* fread() returns 0 on error, so it is OK to store this in a png_size_t
    * instead of an int, which is what fread() actually returns.
    */
   check = fread(data, 1, length, (png_FILE_p)png_get_io_ptr(png_ptr));

   if (check != length)
      png_error(png_ptr, "Read Error");
}

bool ImagePNG::load(std::string filename)
{
    m_bLoaded = 0;

    if (!sofa::helper::system::DataRepository.findFile(filename))
    {
        msg_error("ImagePNG") << "File '" << filename << "' not found ";
        return false;
    }
    FILE *file;
    /* make sure the file is there and open it read-only (binary) */
    if ((file = fopen(filename.c_str(), "rb")) == NULL)
    {
        msg_error("ImagePNG") << "File not found : '" << filename << "'";
        return false;
    }

    png_structp PNG_reader = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (PNG_reader == NULL)
    {
        msg_error("ImagePNG") << "png_create_read_struct failed for file '"<< filename << "'";
        fclose(file);
        return false;
    }

    png_infop PNG_info = png_create_info_struct(PNG_reader);
    if (PNG_info == NULL)
    {
        msg_error("ImagePNG") << "png_create_info_struct failed for file '" << filename << "'";
        png_destroy_read_struct(&PNG_reader, NULL, NULL);
        fclose(file);
        return false;
    }

    png_infop PNG_end_info = png_create_info_struct(PNG_reader);
    if (PNG_end_info == NULL)
    {
        msg_error("ImagePNG") << "png_create_info_struct failed for file '" << filename << "'";
        png_destroy_read_struct(&PNG_reader, &PNG_info, NULL);
        fclose(file);
        return false;
    }

    if (setjmp(png_jmpbuf(PNG_reader)))
    {
        msg_error("ImagePNG") << "Loading failed for PNG file '" << filename << "'";
        png_destroy_read_struct(&PNG_reader, &PNG_info, &PNG_end_info);
        fclose(file);
        return false;
    }

    //png_init_io(PNG_reader, file);
    png_set_read_fn(PNG_reader, file, png_my_read_data);

    png_read_info(PNG_reader, PNG_info);

    png_uint_32 width, height;
    width = png_get_image_width(PNG_reader, PNG_info);
    height = png_get_image_height(PNG_reader, PNG_info);

    png_uint_32 bit_depth, channels, color_type;
    bit_depth = png_get_bit_depth(PNG_reader, PNG_info);
    channels = png_get_channels(PNG_reader, PNG_info);
    color_type = png_get_color_type(PNG_reader, PNG_info);

#ifndef NDEBUG
    msg_info("ImagePNG") << " "<<filename<<": "<<width<<"x"<<height<<"x"<<bit_depth*channels;
#endif
    bool changed = false;
    if (color_type == PNG_COLOR_TYPE_PALETTE)
    {
        png_set_palette_to_rgb(PNG_reader);
        changed = true;
    }

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    {
#ifdef RECENT_LIBPNG
        png_set_expand_gray_1_2_4_to_8(PNG_reader);
#else
        png_set_gray_1_2_4_to_8(PNG_reader);    // deprecated from libpng 1.2.9
#endif
        changed = true;
    }

    if (bit_depth == 16)
    {
        png_set_strip_16(PNG_reader);
        changed = true;
    }
    if (changed)
    {
        png_read_update_info(PNG_reader, PNG_info);
        bit_depth = png_get_bit_depth(PNG_reader, PNG_info);
        channels = png_get_channels(PNG_reader, PNG_info);
//        color_type = png_get_color_type(PNG_reader, PNG_info);

#ifndef NDEBUG
        msg_info("ImagePNG") << "Converted PNG image "<<filename<<": "<<width<<"x"<<height<<"x"<<bit_depth*channels;
#endif
    }

    Image::DataType dataType;
    Image::ChannelFormat channelFormat;
    switch (bit_depth)
    {
    case 8:
        dataType = Image::UNORM8;
        break;
    default:
        msg_error("ImagePNG") << "File '" << filename << "' has unsupported bit depth: " << bit_depth ;
        return false;
    }
    switch (channels)
    {
    case 1:
        channelFormat = Image::L;
        break;
    case 2:
        channelFormat = Image::LA;
        break;
    case 3:
        channelFormat = Image::RGB;
        break;
    case 4:
        channelFormat = Image::RGBA;
        break;
    default:
        msg_error("ImagePNG") << "in '" << filename << "', unsupported number of channels: " << channels ;
        return false;
    }

    init(width, height, 1, 1, dataType, channelFormat);
    png_byte** PNG_rows = (png_byte**)malloc(height * sizeof(png_byte*));

    unsigned char *data = getPixels();
    unsigned lineSize = getLineSize(0);
    for (png_uint_32 row = 0; row < height; ++row)
        PNG_rows[height - 1 - row] = data+row*lineSize;

    png_read_image(PNG_reader, PNG_rows);

    free(PNG_rows);

    png_read_end(PNG_reader, PNG_end_info);

    png_destroy_read_struct(&PNG_reader, &PNG_info, &PNG_end_info);
    fclose(file);
    m_bLoaded = 1;
    return true;
}

// we have to give our own writing function to libpng in order to use the "fwrite" function belonging to Sofa
// and not the one belonging to libpng since Sofa do the "fopen" on the FILE struct
void png_my_write_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    png_size_t check;

    if (png_ptr == NULL)
        return;

    check = fwrite(data, 1, length, (png_FILE_p)png_get_io_ptr(png_ptr));

    if (check != length)
        png_error(png_ptr, "Write Error");
}

void png_default_flush(png_structp png_ptr)
{
    png_FILE_p io_ptr;

    if (png_ptr == NULL)
        return;

    io_ptr = (png_FILE_p)(png_get_io_ptr(png_ptr));
    fflush(io_ptr);
}

bool ImagePNG::save(std::string filename, int compression_level)
{

    FILE *file;
#ifndef NDEBUG
    std::cout << "Writing PNG file " << filename << std::endl;
#endif
    /* make sure the file is there and open it read-only (binary) */
    if ((file = fopen(filename.c_str(), "wb")) == NULL)
    {
        msg_error("ImagePNG") << "File write access failed to '" << filename << "'" ;
        return false;
    }

    png_structp PNG_writer = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (PNG_writer == NULL)
    {
        msg_error("ImagePNG") << "png_create_write_struct failed for file '"<< filename << "'";
        fclose(file);
        return false;
    }

    png_infop PNG_info = png_create_info_struct(PNG_writer);
    if (PNG_info == NULL)
    {
        msg_error("ImagePNG") << "png_create_info_struct failed for file '" << filename << "'";
        png_destroy_write_struct(&PNG_writer, NULL);
        fclose(file);
        return false;
    }

    if (setjmp(png_jmpbuf(PNG_writer)))
    {
        msg_error("ImagePNG") << "Writing failed for PNG file '" << filename << "'";
        png_destroy_write_struct(&PNG_writer, &PNG_info);
        fclose(file);
        return false;
    }

    //png_init_io(PNG_writer, file);
    png_set_write_fn(PNG_writer, file, png_my_write_data, png_default_flush);

    png_uint_32 width, height;
    png_uint_32 bit_depth, channels, color_type;

    width = getWidth();
    height = getHeight();

    bit_depth = getBytesPerChannel() * 8;
    channels = getChannelCount();
    if (channels == 1)
        color_type = PNG_COLOR_TYPE_GRAY;
    else if (channels == 2)
        color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
    else if (channels == 3)
        color_type = PNG_COLOR_TYPE_RGB;
    else
        color_type = PNG_COLOR_TYPE_RGB_ALPHA;

    if (bit_depth != 8)
    {
        msg_error("ImagePNG") << "Unsupported bitdepth "<< bit_depth <<" to write to PNG file '"<<filename<<"'";
        png_destroy_write_struct(&PNG_writer, &PNG_info);
        fclose(file);
        return false;
    }
#ifndef NDEBUG
    msg_info("ImagePNG") << "PNG image "<<filename<<": "<<width<<"x"<<height<<"x"<<bit_depth*channels;
#endif
    png_set_IHDR(PNG_writer, PNG_info, width, height,
            bit_depth, color_type, PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    /* set the zlib compression level */
    if (compression_level!=-1)
    {
        if (compression_level>=0 && compression_level<=9)
            png_set_compression_level(PNG_writer, compression_level);
        else
            msg_error("ImagePNG") << "Compression level must be a value between 0 and 9" ;
    }

    png_byte** PNG_rows = (png_byte**)malloc(height * sizeof(png_byte*));

    unsigned char *data = getPixels();
    unsigned lineSize = getLineSize(0);
    for (png_uint_32 row = 0; row < height; ++row)
        PNG_rows[height - 1 - row] = data+row*lineSize;

    png_set_rows(PNG_writer, PNG_info, PNG_rows);

    png_write_png(PNG_writer, PNG_info, PNG_TRANSFORM_IDENTITY, NULL);

    free(PNG_rows);

    png_destroy_write_struct(&PNG_writer, &PNG_info);
    fclose(file);
    return true;
}

#endif

} // namespace io

} // namespace helper

} // namespace sofa

