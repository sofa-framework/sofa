#include "PluginA.h"
#include <sofa/core/Plugin.h>

#include <PluginMonitor/PluginMonitor.h>

static struct PluginAMonitor {
    PluginAMonitor() { PluginA_loaded++; }
    ~PluginAMonitor() { PluginA_unloaded++; }
} PluginAMonitor_;

class PluginA: public sofa::core::Plugin {
  SOFA_PLUGIN(PluginA);
public:
    PluginA(): Plugin("PluginA") {
        addComponent<Foo>("Component Foo");
    }
};

int FooClass = PluginA::registerObject("Component Foo")
.add<Foo>();

int BarClass = PluginA::registerObject("Component Bar")
.add< Bar<float> >(true)
.add< Bar<double> >();

SOFA_PLUGIN_ENTRY_POINT(PluginA);
