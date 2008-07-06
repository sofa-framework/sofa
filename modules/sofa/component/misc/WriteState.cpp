
#include <sofa/component/misc/WriteState.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(WriteState)

using namespace defaulttype;



int WriteStateClass = core::RegisterObject("Write State vectors to file at each timestep")
        .add< WriteState >();




//Create a Write State component each time a mechanical state is found
simulation::Visitor::Result WriteStateCreator::processNodeTopDown( simulation::Node* gnode)
{
    sofa::core::componentmodel::behavior::BaseMechanicalState * mstate = dynamic_cast<sofa::core::componentmodel::behavior::BaseMechanicalState *>( gnode->getMechanicalState());
    if (!mstate)   return simulation::Visitor::RESULT_CONTINUE;
    //We have a mechanical state
    addWriteState(mstate, gnode);
    return simulation::Visitor::RESULT_CONTINUE;
}


void WriteStateCreator::addWriteState(sofa::core::componentmodel::behavior::BaseMechanicalState *ms, simulation::Node* gnode)
{
    sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping;
    context->get(mapping);
    if ( createInMapping || mapping == NULL)
    {
        sofa::component::misc::WriteState *ws;
        context->get(ws);
        if ( ws == NULL )
        {
            ws = new sofa::component::misc::WriteState(); gnode->addObject(ws);
        }

        std::ostringstream ofilename;
        ofilename << sceneName << "_" << counterWriteState << "_" << ms->getName()  << "_mstate.txt" ;

        ws->f_filename.setValue(ofilename.str()); ws->init(); ws->f_listening.setValue(true);  //Activated at init

        ++counterWriteState;
    }
}



//if state is true, we activate all the write states present in the scene.
simulation::Visitor::Result WriteStateActivator::processNodeTopDown( simulation::Node* gnode)
{
    sofa::component::misc::WriteState *ws = gnode->get< sofa::component::misc::WriteState >();
    if (ws != NULL) { changeStateWriter(ws);}
    return simulation::Visitor::RESULT_CONTINUE;
}

void WriteStateActivator::changeStateWriter(sofa::component::misc::WriteState*ws)
{
    if (!state) ws->reset();
    ws->f_listening.setValue(state);
}






} // namespace misc

} // namespace component

} // namespace sofa
