#include <image/config.h>

#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa::component
{


extern "C" {
    SOFA_EXPORT_DYNAMIC_LIBRARY void initExternalModule();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleName();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleVersion();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleLicense();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleDescription();
    SOFA_EXPORT_DYNAMIC_LIBRARY const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;
        msg_warning("CImgPlugin") << "CImgPlugin has been deprecated, and has been merged into the image plugin. Please use the image plugin directly.";

        sofa::helper::system::PluginManager::getInstance().loadPlugin("image");
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
    return "";
}

}
