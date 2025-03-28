#include <BeamPlastic/init.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace beamplastic
{

namespace forcefield
{

extern void registerBeamPlasticFEMForceField(sofa::core::ObjectFactory* factory);

}

extern "C" {
    BEAMPLASTIC_API void initExternalModule();
    BEAMPLASTIC_API const char* getModuleName();
    BEAMPLASTIC_API const char* getModuleVersion();
    BEAMPLASTIC_API const char* getModuleLicense();
    BEAMPLASTIC_API const char* getModuleDescription();
    BEAMPLASTIC_API void registerObjects(sofa::core::ObjectFactory* factory);
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
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    forcefield::registerBeamPlasticFEMForceField(factory);
}

} // namespace beamplastic
