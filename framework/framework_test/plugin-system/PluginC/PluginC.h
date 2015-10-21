#ifndef PLUGINC_H
#define PLUGINC_H

#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_PLUGINC
#define SOFA_PluginC_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_PluginC_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

void SOFA_PluginC_API PluginC_function();

#endif
