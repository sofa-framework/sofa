#include <BeamPlastic/config.h>

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;

namespace sofa
{

namespace component
{

extern "C" {
    SOFA_BeamPlastic_API void initExternalModule();
    SOFA_BeamPlastic_API const char* getModuleName();
    SOFA_BeamPlastic_API const char* getModuleVersion();
    SOFA_BeamPlastic_API const char* getModuleLicense();
    SOFA_BeamPlastic_API const char* getModuleDescription();
    SOFA_BeamPlastic_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
    // initialisation code, executed before any component is created
}

const char* getModuleName()
{
    return "BeamPlastic";
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
    return "This plugin provides all necessary tools for stent expansion simulation";
    //TODO: complete description
}

const char* getModuleComponentList()
{
    //TODO: Comma-separated list of the components in this plugin, empty for now
    return "BeamPlasticFEMForceField";
}

} //component
} //sofa