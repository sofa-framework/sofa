#include "ScriptController.h"
//#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace controller
{



ScriptController::ScriptController()
    : Controller()
{
    // various initialization stuff here...
    f_listening = true; // par défaut, on écoute les events sinon le script va pas servir à grand chose
}



void ScriptController::parse(sofa::core::objectmodel::BaseObjectDescription *arg)
{
    Controller::parse(arg);

    std::cout<<getName()<<" ScriptController::parse"<<std::endl;

    // load & bind script
    loadScript();
    // call script notifications...
    script_onLoaded( dynamic_cast<simulation::tree::GNode*>(getContext()) );
    script_createGraph( dynamic_cast<simulation::tree::GNode*>(getContext()) );
}

void ScriptController::init()
{
    Controller::init();
    // init the script
    script_initGraph( dynamic_cast<simulation::tree::GNode*>(getContext()) );
}

void ScriptController::storeResetState()
{
    Controller::storeResetState();
    // init the script
    script_storeResetState();
}

void ScriptController::reset()
{
    Controller::reset();
    // init the script
    script_reset();
}

void ScriptController::cleanup()
{
    Controller::cleanup();
    // init the script
    script_cleanup();
}

void ScriptController::onBeginAnimationStep(const double dt)
{
    script_onBeginAnimationStep(dt);
}

} // namespace controller

} // namespace component

} // namespace sofa



