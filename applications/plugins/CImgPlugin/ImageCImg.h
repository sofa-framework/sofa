#ifndef SOFA_HELPER_IO_IMAGECIMG_H
#define SOFA_HELPER_IO_IMAGECIMG_H

#include <CImgPlugin.h>

#include <sofa/helper/io/Image.h>
#include <string>
#include <cassert>
#include <vector>

#include <sofa/helper/system/config.h>
#include <sofa/helper/logging/MessageDispatcher.h>
#include <sofa/helper/logging/CountingMessageHandler.h>
#include <sofa/helper/logging/LoggingMessageHandler.h>

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
    ImageCImg ()
    {
        helper::logging::MessageDispatcher::clearHandlers() ;
        helper::logging::MessageDispatcher::addHandler( &helper::logging::MainCountingMessageHandler::getInstance() ) ;
        helper::logging::MessageDispatcher::addHandler( &helper::logging::MainLoggingMessageHandler::getInstance() ) ;
    }

    ImageCImg (const std::string &filename)
    {
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
