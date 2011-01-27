#ifndef INITPluginExample_H
#define INITPluginExample_H


#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_PLUGINEXAMPLE
#define SOFA_PluginExample_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_PluginExample_API  SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#endif // INITPluginExample_H
