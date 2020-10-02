#include "ScriptDataEngine.h"

namespace sofa
{

namespace component
{

namespace controller
{

void ScriptDataEngine::parse(sofa::core::objectmodel::BaseObjectDescription *arg)
{
    Inherit1::parse(arg);

    // load & bind script
    loadScript();
    // call script notifications...
    //script_onLoaded( down_cast<simulation::Node>(getContext()) );
    //script_createGraph( down_cast<simulation::Node>(getContext()) );
}

ScriptDataEngine::ScriptDataEngine() : Inherit1()
{
    f_listening = true;
}

ScriptDataEngine::~ScriptDataEngine()
{

}

void ScriptDataEngine::call_update()
{
    update();
}

void ScriptDataEngine::doUpdate()
{
    script_update();
}

void ScriptDataEngine::init()
{
    Inherit1::init();
}

void ScriptDataEngine::reinit()
{
}

} // sofa
} // component
} // controller

