#include <iostream>

#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_PLUGINMONITOR
# define SOFA_PLUGINMONITOR_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
# define SOFA_PLUGINMONITOR_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

extern int SOFA_PLUGINMONITOR_API PluginA_loaded;
extern int SOFA_PLUGINMONITOR_API PluginA_unloaded;
extern int SOFA_PLUGINMONITOR_API PluginB_loaded;
extern int SOFA_PLUGINMONITOR_API PluginB_unloaded;
extern int SOFA_PLUGINMONITOR_API PluginC_loaded;
extern int SOFA_PLUGINMONITOR_API PluginC_unloaded;
extern int SOFA_PLUGINMONITOR_API PluginD_loaded;
extern int SOFA_PLUGINMONITOR_API PluginD_unloaded;
extern int SOFA_PLUGINMONITOR_API PluginE_loaded;
extern int SOFA_PLUGINMONITOR_API PluginE_unloaded;
extern int SOFA_PLUGINMONITOR_API PluginF_loaded;
extern int SOFA_PLUGINMONITOR_API PluginF_unloaded;

void SOFA_PLUGINMONITOR_API reset_plugin_monitor();
