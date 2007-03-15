/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/helper/io/ImagePNG.h>
#include <iostream>

#ifdef SOFA_HAVE_PNG
#include <png.h>
#ifdef _MSC_VER
#pragma comment(lib,"libpng.lib")
#pragma comment(lib,"zlib.lib")
#endif
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

Creator<Image::Factory,ImagePNG> ImagePNGClass("png");

bool ImagePNG::load(const std::string &filename)
{
    FILE *file;
    /* make sure the file is there and open it read-only (binary) */
    if ((file = fopen(filename.c_str(), "rb")) == NULL)
    {
        std::cerr << "File not found : " << filename << std::endl;
        return false;
    }

    png_structp PNG_reader = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (PNG_reader == NULL)
    {
        std::cerr << "png_create_read_struct failed for file "<< filename << std::endl;
        fclose(file);
        return false;
    }

    png_infop PNG_info = png_create_info_struct(PNG_reader);
    if (PNG_info == NULL)
    {
        std::cerr << "png_create_info_struct failed for file " << filename << std::endl;
        png_destroy_read_struct(&PNG_reader, NULL, NULL);
        fclose(file);
        return false;
    }

    png_infop PNG_end_info = png_create_info_struct(PNG_reader);
    if (PNG_end_info == NULL)
    {
        std::cerr << "png_create_info_struct failed for file " << filename << std::endl;
        png_destroy_read_struct(&PNG_reader, &PNG_info, NULL);
        fclose(file);
        return false;
    }

    if (setjmp(png_jmpbuf(PNG_reader)))
    {
        std::cerr << "Loading failed for PNG file " << filename << std::endl;
        png_destroy_read_struct(&PNG_reader, &PNG_info, &PNG_end_info);
        fclose(file);
        return false;
    }

    png_init_io(PNG_reader, file);

    png_read_info(PNG_reader, PNG_info);

    png_uint_32 width, height;
    width = png_get_image_width(PNG_reader, PNG_info);
    height = png_get_image_height(PNG_reader, PNG_info);

    png_uint_32 bit_depth, channels, color_type;
    bit_depth = png_get_bit_depth(PNG_reader, PNG_info);
    channels = png_get_channels(PNG_reader, PNG_info);
    color_type = png_get_color_type(PNG_reader, PNG_info);

    std::cout << "PNG image "<<filename<<": "<<width<<"x"<<height<<"x"<<bit_depth*channels<<std::endl;
    bool changed = false;
    if (color_type == PNG_COLOR_TYPE_PALETTE)
    {
        png_set_palette_to_rgb(PNG_reader);
        changed = true;
    }

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    {
        png_set_gray_1_2_4_to_8(PNG_reader);
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
        color_type = png_get_color_type(PNG_reader, PNG_info);
        std::cout << "Converted PNG image "<<filename<<": "<<width<<"x"<<height<<"x"<<bit_depth*channels<<std::endl;
    }
    init(width, height, channels*bit_depth);
    png_byte** PNG_rows = (png_byte**)malloc(height * sizeof(png_byte*));

    for (png_uint_32 row = 0; row < height; ++row)
        PNG_rows[height - 1 - row] = getData()+row*getLineSize();

    png_read_image(PNG_reader, PNG_rows);

    free(PNG_rows);

    png_read_end(PNG_reader, PNG_end_info);

    png_destroy_read_struct(&PNG_reader, &PNG_info, &PNG_end_info);
    fclose(file);

    return true;
}

bool ImagePNG::save(const std::string& filename)
{

    FILE *file;
    std::cout << "Writing PNG file " << filename << std::endl;
    /* make sure the file is there and open it read-only (binary) */
    if ((file = fopen(filename.c_str(), "wb")) == NULL)
    {
        std::cerr << "File write access failed : " << filename << std::endl;
        return false;
    }

    png_structp PNG_writer = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (PNG_writer == NULL)
    {
        std::cerr << "png_create_write_struct failed for file "<< filename << std::endl;
        fclose(file);
        return false;
    }

    png_infop PNG_info = png_create_info_struct(PNG_writer);
    if (PNG_info == NULL)
    {
        std::cerr << "png_create_info_struct failed for file " << filename << std::endl;
        png_destroy_write_struct(&PNG_writer, NULL);
        fclose(file);
        return false;
    }

    if (setjmp(png_jmpbuf(PNG_writer)))
    {
        std::cerr << "Writing failed for PNG file " << filename << std::endl;
        png_destroy_write_struct(&PNG_writer, &PNG_info);
        fclose(file);
        return false;
    }

    png_init_io(PNG_writer, file);

    png_uint_32 width, height;
    png_uint_32 bit_depth, channels, color_type;

    width = getWidth();
    height = getHeight();

    bit_depth = (getNbBits()<8)?getNbBits():8;
    channels = (getNbBits()+7)/8;
    if (channels == 1)
        color_type = PNG_COLOR_TYPE_GRAY;
    else if (channels == 2)
        color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
    else if (channels == 3)
        color_type = PNG_COLOR_TYPE_RGB;
    else if (channels == 4)
        color_type = PNG_COLOR_TYPE_RGB_ALPHA;
    else
    {
        std::cerr << "Unsupported bitdepth "<<getNbBits()<<" to write to PNG file "<<filename<<std::endl;
        png_destroy_write_struct(&PNG_writer, &PNG_info);
        fclose(file);
        return false;
    }
    std::cout << "PNG image "<<filename<<": "<<width<<"x"<<height<<"x"<<bit_depth*channels<<std::endl;
    png_set_IHDR(PNG_writer, PNG_info, width, height,
            bit_depth, color_type, PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    /* set the zlib compression level */
    //png_set_compression_level(PNG_writer, Z_BEST_COMPRESSION);

    png_byte** PNG_rows = (png_byte**)malloc(height * sizeof(png_byte*));

    for (png_uint_32 row = 0; row < height; ++row)
        PNG_rows[height - 1 - row] = getData()+row*getLineSize();

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

