#include "PluginB.h"
#include <PluginMonitor/PluginMonitor.h>

static struct PluginBMonitor {
    PluginBMonitor() { PluginB_loaded++; }
    ~PluginBMonitor() { PluginB_unloaded++; }
} PluginBMonitor_;

extern "C" {

void initExternalModule()
{
}

const char* getModuleName()
{
    return "PluginB";
}

const char* getModuleVersion()
{
    return "1.0";
}

const char* getModuleLicense()
{
    return "None";
}

const char* getModuleDescription()
{
    return "This is a empty, old-style, plugin";
}

const char* getModuleComponentList()
{
    return "";
}

}
