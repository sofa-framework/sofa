#ifndef SOFA_HELPER_IO_IMAGECIMG_H
#define SOFA_HELPER_IO_IMAGECIMG_H

#include <memory>
#include <string>
#include <vector>
#include <sofa/helper/io/Image.h>
#include <CImgPlugin/CImgPlugin.h>


namespace sofa::helper::io
{

class SOFA_CIMGPLUGIN_API ImageCImgCreators
{

    std::vector<std::shared_ptr<sofa::helper::io::Image::FactoryImage::Creator>> creators;
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

} // namespace sofa::helper::io


#endif // SOFA_HELPER_IO_IMAGECIMG_H
