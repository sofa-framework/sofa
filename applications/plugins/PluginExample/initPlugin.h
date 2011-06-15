#ifndef INITMyPluginExample_H
#define INITMyPluginExample_H


#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_PLUGINEXAMPLE
#define SOFA_MyPluginExample_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_MyPluginExample_API  SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

/** \mainpage
  This is a simple example plugin.
  */

#endif // INITMyPluginExample_H
