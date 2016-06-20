#include "PluginMonitor.h"

int SOFA_PLUGINMONITOR_API PluginA_loaded;
int SOFA_PLUGINMONITOR_API PluginA_unloaded;
int SOFA_PLUGINMONITOR_API PluginB_loaded;
int SOFA_PLUGINMONITOR_API PluginB_unloaded;
int SOFA_PLUGINMONITOR_API PluginC_loaded;
int SOFA_PLUGINMONITOR_API PluginC_unloaded;
int SOFA_PLUGINMONITOR_API PluginD_loaded;
int SOFA_PLUGINMONITOR_API PluginD_unloaded;
int SOFA_PLUGINMONITOR_API PluginE_loaded;
int SOFA_PLUGINMONITOR_API PluginE_unloaded;
int SOFA_PLUGINMONITOR_API PluginF_loaded;
int SOFA_PLUGINMONITOR_API PluginF_unloaded;

void reset_plugin_monitor() {
    PluginA_loaded = 0;
    PluginA_unloaded = 0;
    PluginB_loaded = 0;
    PluginB_unloaded = 0;
    PluginC_loaded = 0;
    PluginC_unloaded = 0;
    PluginD_loaded = 0;
    PluginD_unloaded = 0;
    PluginE_loaded = 0;
    PluginE_unloaded = 0;
    PluginF_loaded = 0;
    PluginF_unloaded = 0;
}

