#include "PluginF.h"
#include <sofa/core/Plugin.h>

#include <PluginMonitor/PluginMonitor.h>

static struct PluginFMonitor {
    PluginFMonitor() { PluginF_loaded++; }
    ~PluginFMonitor() { PluginF_unloaded++; }
} PluginFMonitor_;

class PluginF: public sofa::core::Plugin {
  SOFA_PLUGIN(PluginF);
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

int FooFClass = PluginF::registerObject("Component FooF")
  .add< FooF >();

int BarFClass = PluginF::registerObject("Component BarF")
  .add< BarF<float> >(true)
  .add< BarF<double> >();

SOFA_PLUGIN_ENTRY_POINT(PluginF);
