#include "PluginF.h"
#include <sofa/core/Plugin.h>

#include <PluginMonitor/PluginMonitor.h>

static struct PluginFMonitor {
    PluginFMonitor() { PluginF_loaded++; }
    ~PluginFMonitor() { PluginF_unloaded++; }
} PluginFMonitor_;

class PluginF: public sofa::core::Plugin {
public:
    PluginF(): Plugin("PluginF") {
        addComponent<FooF>("Component FooF");
        addComponent< BarF<float> >("Component BarF");
        addTemplateInstance< BarF<double> >();
    }

    virtual bool canBeUnloaded() {
        return false;
    }
};

SOFA_PLUGIN(PluginF);
