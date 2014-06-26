#include <sofa/core/Plugin.h>

#include <sofa/core/objectmodel/BaseObject.h>

#include <PluginMonitor/PluginMonitor.h>
#include <PluginA/PluginA.h>


static struct PluginDMonitor {
    PluginDMonitor() { PluginD_loaded++; }
    ~PluginDMonitor() { PluginD_unloaded++; }
} PluginDMonitor_;


class FooD: public Foo {
};

template <class C>
class BazD: public sofa::core::objectmodel::BaseObject {
};

class VecD {
};


class PluginD: public sofa::core::Plugin {
  SOFA_PLUGIN(PluginD);
public:
    PluginD(): Plugin("PluginD") {
    }
};

int FooDClass = PluginD::registerObject("Component FooD")
  .add<FooD>();

int BarClass = PluginD::registerObject("")
.add< Bar<VecD> >();

int BazDClass = PluginD::registerObject("Component BazD")
.add< BazD<float> >(true)
.add< BazD<double> >();

SOFA_PLUGIN_ENTRY_POINT(PluginD);
