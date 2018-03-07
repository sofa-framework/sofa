#include <iostream>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/logging/Messaging.h>

#include "ImageCImg.h"
#include "SOFACImg.h"

MSG_REGISTER_CLASS(sofa::helper::io::ImageCImg, "ImageCImg")

namespace sofa
{

namespace helper
{


namespace io
{

SOFA_DECL_CLASS(ImageCImg)

std::vector<std::string> ImageCImgCreators::cimgSupportedExtensions {
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

ImageCImgCreators::ImageCImgCreators()
{
    for(unsigned int i=0 ; i<cimgSupportedExtensions.size() ; i++)
    {
        const std::string& ext = cimgSupportedExtensions[i];
        if (!sofa::helper::io::Image::FactoryImage::HasKey(ext))
        {
            creators.push_back(new Creator<helper::io::Image::FactoryImage, ImageCImg>(ext));
        }
    }

}

void ImageCImg::setCimgCreators()
{
    static ImageCImgCreators cimgCreators;
}

bool ImageCImg::load(std::string filename)
{
    cimg_library::cimg::exception_mode(0);
    m_bLoaded = 0;

    if (!sofa::helper::system::DataRepository.findFile(filename))
    {
        msg_error() << "File " << filename << " not found ";
        return false;
    }

    cimg_library::CImg<unsigned char> cimgImage;
    try
    {
        cimgImage.load(filename.c_str());
    }
    catch(cimg_library::CImgIOException e)
    {
        msg_error() << "Caught exception while loading: " << e.what();
        return false;
    }

    unsigned int width, height, channels;
    width = cimgImage.width();
    height = cimgImage.height();
    channels = cimgImage.spectrum();

    Image::DataType dataType;
    Image::ChannelFormat channelFormat;

    if (!cimg_library::cimg::strcasecmp(cimgImage.pixel_type(),"unsigned char"))
    {
        dataType = Image::UNORM8;
    }
    else
    {
        msg_error() << "in " << filename << ", unsupported bit depth: " << cimgImage.pixel_type();
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
        msg_error() << "in " << filename << ", unsupported number of channels: " << channels;
        return false;
    }

    //flip image on Y axis
    //Cimg top to bottom, Sofa bottom to top
    cimgImage.mirror("y");

    init(width, height, 1, 1, dataType, channelFormat);
    unsigned char *data = getPixels();

    // CImg stores channel non-interleaved
    // e.g R1R2R3...G1G2G3....B1B2B3
    // sofa::Image stores it interleaved
    // e.g R1G1B1R2G2B2R3G3B3
    unsigned int totalSize = width * height;

    for(unsigned int xy=0 ; xy < totalSize ; xy++)
            for(unsigned int c=0 ; c < channels ; c++)
                data[xy * channels + c] = cimgImage[xy + c*totalSize];


    m_bLoaded = 1;
    return true;
}

//compression_level seems not to be usable with Cimg
bool ImageCImg::save(std::string filename, int /* compression_level */)
{
    bool res = false;
    std::string ext;
    //check extension
    for(size_t i=0 ; i<ImageCImgCreators::cimgSupportedExtensions.size() ; i++)
    {
        const std::string currExt = ImageCImgCreators::cimgSupportedExtensions[i];
        if(filename.substr(filename.find_last_of(".") + 1) == currExt)
            ext = currExt;
    }

    const unsigned char *data = getPixels();
    unsigned int totalSize = getWidth() * getHeight();
    unsigned int channelsNb = this->getChannelCount();

    cimg_library::CImg<unsigned char> cimgImage(getWidth(), getHeight(),1, channelsNb);

    try
    {
        for(unsigned int xy=0 ; xy < totalSize ; xy++)
                for(unsigned int c=0 ; c < channelsNb ; c++)
                    cimgImage[xy + c*totalSize] = data[xy * channelsNb + c];
    }
    catch (cimg_library::CImgIOException e)
    {
        msg_error() << "Caught exception while saving: " << e.what();
        res = false;
    }

    //flip image on Y axis
    //Cimg top to bottom, Sofa bottom to top
    cimgImage.mirror("y");

    try
    {
        if(ext.empty())
        {
            msg_error() << "Cannot recognize extension or file format not supported,"
                                   << "image will be saved as a PNG file.";
            cimgImage.save_png(filename.c_str());
        }
        else
            cimgImage.save(filename.c_str());

        res = true;
    }
    catch(cimg_library::CImgIOException e)
    {
        msg_error() << "Caught exception while saving: " << e.what();
        res = false;
    }

    return res;
}

} // namespace io

} // namespace helper

} // namespace sofa
