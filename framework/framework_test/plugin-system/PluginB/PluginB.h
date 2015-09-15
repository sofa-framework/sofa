#ifndef PLUGINB_H
#define PLUGINB_H

#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_PLUGINB
#define SOFA_PluginB_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_PluginB_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

extern "C" {

SOFA_PluginB_API void initExternalModule();

SOFA_PluginB_API const char* getModuleName();

SOFA_PluginB_API const char* getModuleVersion();

SOFA_PluginB_API const char* getModuleLicense();

SOFA_PluginB_API const char* getModuleDescription();

SOFA_PluginB_API const char* getModuleComponentList();

}

#endif
