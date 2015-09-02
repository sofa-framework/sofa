
#include <ExternalBehaviorModel/config.h>

namespace sofa
{

namespace component
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_ExternalBehaviorModel_API void initExternalModule();
    SOFA_ExternalBehaviorModel_API const char* getModuleName();
    SOFA_ExternalBehaviorModel_API const char* getModuleVersion();
    SOFA_ExternalBehaviorModel_API const char* getModuleLicense();
    SOFA_ExternalBehaviorModel_API const char* getModuleDescription();
    SOFA_ExternalBehaviorModel_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

const char* getModuleName()
{
    return "ExternalBehaviorModel";
}

const char* getModuleVersion()
{
    return "0.0";
}

const char* getModuleLicense()
{
    return "???";
}


const char* getModuleDescription()
{
    return "Plug an external behavior model with sofa including implicit interaction";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    return  "FEMGridBehaviorModel";
}
}
}

/// Use the SOFA_LINK_CLASS macro for each class, to enable linking on all platforms


SOFA_LINK_CLASS(FEMGridBehaviorModel)



