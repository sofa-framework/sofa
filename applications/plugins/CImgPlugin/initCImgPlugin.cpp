#include "CImgPlugin.h"

#include <ImageCImg.h>

#include <sofa/helper/Factory.h>
#include <sofa/helper/io/Image.h>

#include <iostream>
#include <vector>

#include "SOFACImg.h"

/// CImg is a header only library. The consequence is that if compiled in multiple libraries
/// it is unclear where the "main" structure holding the CImg internal mutexes should be.
/// This generates either link time failure as well as problem with the static initialization
/// of the mutex resulting in run-time crashes.
/// To fix this...we are forcing the Mutex_attr function to be defined in the CImgPlugin only
/// and export it.
namespace cimg_library
{
namespace cimg
{
    extern "C" {
        SOFA_CIMGPLUGIN_API Mutex_info& Mutex_attr() { static Mutex_info val; return val; }
    }
}
}

namespace sofa
{

namespace component
{


extern "C" {
    cimg_library::cimg::Mutex_info& tmp = cimg_library::cimg::Mutex_attr() ;

    SOFA_CIMGPLUGIN_API void initExternalModule();
    SOFA_CIMGPLUGIN_API const char* getModuleName();
    SOFA_CIMGPLUGIN_API const char* getModuleVersion();
    SOFA_CIMGPLUGIN_API const char* getModuleLicense();
    SOFA_CIMGPLUGIN_API const char* getModuleDescription();
    SOFA_CIMGPLUGIN_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;

        sofa::helper::io::ImageCImg::setCimgCreators();
    }
}

const char* getModuleName()
{
    return "CImgPlugin";
}

const char* getModuleVersion()
{
    return "1.0";
}

const char* getModuleLicense()
{
    return "LGPL";
}


const char* getModuleDescription()
{
    return "This plugin can load different kind of images, supported by the CImg toolkit.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    return "ImageCImg";
}

}

}

/// Use the SOFA_LINK_CLASS macro for each class, to enable linking on all platforms
SOFA_LINK_CLASS(ImageCImg)


