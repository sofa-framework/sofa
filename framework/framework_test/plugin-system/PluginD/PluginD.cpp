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
public:
    PluginD(): Plugin("PluginD") {
        addComponent<FooD>("Component Foo");
        addTemplateInstance< Bar<VecD> >();
        addComponent< BazD<float> >();
        addTemplateInstance< BazD<double> >();
    }
};

SOFA_PLUGIN(PluginD);
