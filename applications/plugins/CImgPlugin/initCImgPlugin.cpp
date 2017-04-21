#include "CImgPlugin.h"

#include <ImageCImg.h>

#include <sofa/helper/Factory.h>
#include <sofa/helper/io/Image.h>

#include <iostream>
#include <vector>

namespace sofa
{

namespace component
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
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
    return "0.1";
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


