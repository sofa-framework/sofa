#include "PluginC.h"
#include <sofa/core/Plugin.h>

#include <sofa/core/objectmodel/BaseObject.h>

#include <PluginMonitor/PluginMonitor.h>
#include <PluginA/PluginA.h>


static struct PluginCMonitor {
    PluginCMonitor() { PluginC_loaded++; }
    ~PluginCMonitor() { PluginC_unloaded++; }
} PluginCMonitor_;

class PluginC: public sofa::core::Plugin {
public:
    PluginC(): Plugin("PluginC") {
    }
};

SOFA_PLUGIN(PluginC);


void SOFA_PluginC_API PluginC_function()
{

}
