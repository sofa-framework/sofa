#ifndef SOFA_HELPER_IO_IMAGECIMG_H
#define SOFA_HELPER_IO_IMAGECIMG_H

#include <string>
#include <vector>
#include <sofa/helper/io/Image.h>
#include <CImgPlugin.h>

namespace sofa
{

namespace helper
{

namespace io
{

class SOFA_CIMGPLUGIN_API ImageCImgCreators
{
    std::vector<sofa::helper::io::Image::FactoryImage::Creator*> creators;
public:
    static std::vector<std::string> cimgSupportedExtensions;

    ImageCImgCreators();

};

class SOFA_CIMGPLUGIN_API ImageCImg : public Image
{
public:
    ImageCImg (){}

    ImageCImg (const std::string &filename)
    {
        if(!filename.empty())
            load(filename);
    }

    static void setCimgCreators();

    bool load(std::string filename);
    bool save(std::string filename, int compression_level = -1);
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif // SOFA_HELPER_IO_IMAGECIMG_H
