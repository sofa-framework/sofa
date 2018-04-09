#include "ScriptDataEngine.h"

namespace sofa
{

namespace component
{

namespace controller
{

void ScriptDataEngine::parse(sofa::core::objectmodel::BaseObjectDescription *arg)
{
    DataEngine::parse(arg);

    // load & bind script
    loadScript();
    // call script notifications...
    //script_onLoaded( down_cast<simulation::Node>(getContext()) );
    //script_createGraph( down_cast<simulation::Node>(getContext()) );
}

ScriptDataEngine::ScriptDataEngine() : DataEngine()
{
    f_listening = true;
}

ScriptDataEngine::~ScriptDataEngine()
{

}

void ScriptDataEngine::update()
{
    // DataEngine::update(); doesn't make sense, probably?
    cleanDirty();
    script_update();
}

void ScriptDataEngine::init()
{
    DataEngine::init();
}

void ScriptDataEngine::reinit()
{
}

} // sofa
} // component
} // controller

