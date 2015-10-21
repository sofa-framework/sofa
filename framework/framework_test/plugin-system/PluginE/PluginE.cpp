#include <sofa/core/Plugin.h>

#include <sofa/core/objectmodel/BaseObject.h>

#include <PluginMonitor/PluginMonitor.h>
#include <PluginC/PluginC.h>


static struct PluginEMonitor {
    PluginEMonitor() { PluginE_loaded++; }
    ~PluginEMonitor() { PluginE_unloaded++; }
} PluginEMonitor_;


class PluginE: public sofa::core::Plugin {
  SOFA_PLUGIN(PluginE);
public:
    PluginE(): Plugin("PluginE") {
        PluginC_function();
    }
};

SOFA_PLUGIN_ENTRY_POINT(PluginE);
