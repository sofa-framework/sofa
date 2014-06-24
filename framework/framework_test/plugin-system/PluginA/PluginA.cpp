#include "PluginA.h"
#include <sofa/core/Plugin.h>

#include <PluginMonitor/PluginMonitor.h>

static struct PluginAMonitor {
    PluginAMonitor() { PluginA_loaded++; }
    ~PluginAMonitor() { PluginA_unloaded++; }
} PluginAMonitor_;

class PluginA: public sofa::core::Plugin {
public:
    PluginA(): Plugin("PluginA") {
        addComponent<Foo>("Component Foo");
        addComponent< Bar<float> >("Component Bar");
        addTemplateInstance< Bar<double> >();
    }
};

SOFA_PLUGIN(PluginA);
