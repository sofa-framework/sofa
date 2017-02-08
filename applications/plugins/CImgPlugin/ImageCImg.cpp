#include <ImageCImg.h>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/logging/Messaging.h>

#define cimg_display 0
#include <CImg/SOFACImg.h>

#include <iostream>

namespace sofa
{

namespace helper
{

namespace io
{

SOFA_DECL_CLASS(ImageCImg)

//#ifdef CIMGPLUGIN_HAVE_PNG
//Creator<Image::FactoryImage, ImageCImg> ImageCImg_PNGClass("png");
//#endif // CIMGPLUGIN_HAVE_PNG

//#ifdef CIMGPLUGIN_HAVE_JPEG
//Creator<Image::FactoryImage, ImageCImg> ImageCImg_JPEGClass("jpeg");
//Creator<Image::FactoryImage, ImageCImg> ImageCImg_JPGClass("jpg");
//#endif // CIMGPLUGIN_HAVE_JPEG

//#ifdef CIMGPLUGIN_HAVE_TIFF
//Creator<Image::FactoryImage, ImageCImg> ImageCImg_TIFFClass("tiff");
//Creator<Image::FactoryImage, ImageCImg> ImageCImg_TIFClass("tif");
//#endif // CIMGPLUGIN_HAVE_TIFF


class SOFA_CIMGPLUGIN_API ImageCImgCreators
{
    std::vector<sofa::helper::io::Image::FactoryImage::Creator*> creators;
public:
    ImageCImgCreators()
    {
        std::vector<std::string> cimgSupportedExtensions {
#ifdef CIMGPLUGIN_HAVE_PNG
        "png",
#endif // CIMGPLUGIN_HAVE_PNG
#ifdef CIMGPLUGIN_HAVE_JPEG
        "jpg",
        "jpeg",
#endif // CIMGPLUGIN_HAVE_JPEG
#ifdef CIMGPLUGIN_HAVE_TIFF
        "tif",
        "tiff",
#endif // CIMGPLUGIN_HAVE_TIFF
        "bmp"
        };

        for(unsigned int i=0 ; i<cimgSupportedExtensions.size() ; i++)
        {
            const std::string& ext = cimgSupportedExtensions[i];
            if (!sofa::helper::io::Image::FactoryImage::HasKey(ext))
            {
                creators.push_back(new Creator<helper::io::Image::FactoryImage, ImageCImg>(ext));
            }
        }
    }
};


void ImageCImg::setCimgCreators()
{
    static ImageCImgCreators cimgCreators;
}

bool ImageCImg::load(std::string filename)
{
    //msg_info("ImageCImg") << "Using CImgPlugin for " << filename;

    m_bLoaded = 0;

    if (!sofa::helper::system::DataRepository.findFile(filename))
    {
        msg_error("ImageCImg") << "File " << filename << " not found ";
        return false;
    }

    cimg_library::CImg<unsigned char> image(filename.c_str());

    unsigned int width, height, depth, channels;
    width = image.width();
    height = image.height();
    depth = image.depth();
    channels = image.spectrum();

    Image::DataType dataType;
    Image::ChannelFormat channelFormat;

    if (!cimg_library::cimg::strcasecmp(image.pixel_type(),"unsigned char"))
    {
        dataType = Image::UNORM8;
    }
    else
    {
        msg_error("ImageCImg") << "in " << filename << ", unsupported bit depth: " << image.pixel_type();
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
        msg_error("ImageCImg") << "in " << filename << ", unsupported number of channels: " << channels;
        return false;
    }

    //flip image on X axis
    image.mirror("y");

    init(width, height, 1, 1, dataType, channelFormat);
    unsigned char *data = getPixels();

    // CImg stores channel non-interleaved
    // e.g R1R2R3...G1G2G3....B1B2B3
    // sofa::Image stores it interleaved
    // e.g R1G1B1R2G2B2R3G3B3
    unsigned int totalSize = width * height;

    for(unsigned int xy=0 ; xy < totalSize ; xy++)
            for(unsigned int c=0 ; c < channels ; c++)
                data[xy * channels + c] = image[xy + c*totalSize];


    m_bLoaded = 1;
    return true;
}

//bool ImagePNG::save(std::string filename, int compression_level)
//{

//    FILE *file;
//#ifndef NDEBUG
//    std::cout << "Writing PNG file " << filename << std::endl;
//#endif
//    /* make sure the file is there and open it read-only (binary) */
//    if ((file = fopen(filename.c_str(), "wb")) == NULL)
//    {
//        std::cerr << "File write access failed : " << filename << std::endl;
//        return false;
//    }

//    png_structp PNG_writer = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
//    if (PNG_writer == NULL)
//    {
//        std::cerr << "png_create_write_struct failed for file "<< filename << std::endl;
//        fclose(file);
//        return false;
//    }

//    png_infop PNG_info = png_create_info_struct(PNG_writer);
//    if (PNG_info == NULL)
//    {
//        std::cerr << "png_create_info_struct failed for file " << filename << std::endl;
//        png_destroy_write_struct(&PNG_writer, NULL);
//        fclose(file);
//        return false;
//    }

//    if (setjmp(png_jmpbuf(PNG_writer)))
//    {
//        std::cerr << "Writing failed for PNG file " << filename << std::endl;
//        png_destroy_write_struct(&PNG_writer, &PNG_info);
//        fclose(file);
//        return false;
//    }

//    //png_init_io(PNG_writer, file);
//    png_set_write_fn(PNG_writer, file, png_my_write_data, png_default_flush);

//    png_uint_32 width, height;
//    png_uint_32 bit_depth, channels, color_type;

//    width = getWidth();
//    height = getHeight();

//    bit_depth = getBytesPerChannel() * 8;
//    channels = getChannelCount();
//    if (channels == 1)
//        color_type = PNG_COLOR_TYPE_GRAY;
//    else if (channels == 2)
//        color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
//    else if (channels == 3)
//        color_type = PNG_COLOR_TYPE_RGB;
//    else
//        color_type = PNG_COLOR_TYPE_RGB_ALPHA;

//    if (bit_depth != 8)
//    {
//        std::cerr << "Unsupported bitdepth "<< bit_depth <<" to write to PNG file "<<filename<<std::endl;
//        png_destroy_write_struct(&PNG_writer, &PNG_info);
//        fclose(file);
//        return false;
//    }
//#ifndef NDEBUG
//    std::cout << "PNG image "<<filename<<": "<<width<<"x"<<height<<"x"<<bit_depth*channels<<std::endl;
//#endif
//    png_set_IHDR(PNG_writer, PNG_info, width, height,
//            bit_depth, color_type, PNG_INTERLACE_NONE,
//            PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
//    /* set the zlib compression level */
//    if (compression_level!=-1)
//    {
//        if (compression_level>=0 && compression_level<=9)
//            png_set_compression_level(PNG_writer, compression_level);
//        else
//            std::cerr << "ERROR: compression level must be a value between 0 and 9" << std::endl;
//    }

//    png_byte** PNG_rows = (png_byte**)malloc(height * sizeof(png_byte*));

//    unsigned char *data = getPixels();
//    unsigned lineSize = getLineSize(0);
//    for (png_uint_32 row = 0; row < height; ++row)
//        PNG_rows[height - 1 - row] = data+row*lineSize;

//    png_set_rows(PNG_writer, PNG_info, PNG_rows);

//    png_write_png(PNG_writer, PNG_info, PNG_TRANSFORM_IDENTITY, NULL);

//    free(PNG_rows);

//    png_destroy_write_struct(&PNG_writer, &PNG_info);
//    fclose(file);
//    return true;
//}

} // namespace io

} // namespace helper

} // namespace sofa
