#include <SofaMatrix/imgui/init.h>
#include <sofa/core/ObjectFactory.h>

namespace sofamatriximgui
{

void initializePlugin() 
{
    static bool first = true;
    if (first) {
        first = false;
        // Register components here
    }
}

}

extern "C" 
{
    SOFAMATRIX_IMGUI_API void initExternalModule() 
    {
        sofamatriximgui::initializePlugin();
    }

    SOFAMATRIX_IMGUI_API const char* getModuleName() 
    {
        return sofamatriximgui::MODULE_NAME;
    }

    SOFAMATRIX_IMGUI_API const char* getModuleVersion() 
    {
        return sofamatriximgui::MODULE_VERSION;
    }

    SOFAMATRIX_IMGUI_API const char* getModuleLicense() 
    {
        return "LGPL";
    }

    SOFAMATRIX_IMGUI_API const char* getModuleDescription() 
    {
        return "SOFA plugin for SofaMatrix.imgui";
    }
}
