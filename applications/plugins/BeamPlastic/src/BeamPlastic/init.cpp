#include <BeamPlastic/init.h>

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;

namespace beamplastic
{

extern "C" {
    BEAMPLASTIC_API void initExternalModule();
    BEAMPLASTIC_API const char* getModuleName();
    BEAMPLASTIC_API const char* getModuleVersion();
    BEAMPLASTIC_API const char* getModuleLicense();
    BEAMPLASTIC_API const char* getModuleDescription();
}

void initExternalModule()
{
    init();
}

void init()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

const char* getModuleName()
{
    return MODULE_NAME;
}

const char* getModuleVersion()
{
    return MODULE_VERSION;
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "This plugin provides all necessary tools for stent expansion simulation";
    //TODO: complete description
}

} // namespace beamplastic
