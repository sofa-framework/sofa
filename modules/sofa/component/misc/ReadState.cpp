#include <sofa/component/misc/ReadState.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(ReadState)

using namespace defaulttype;

int ReadStateClass = core::RegisterObject("Read State vectors from file at each timestep")
        .add< ReadState >();


//Create a Read State component each time a mechanical state is found
simulation::Visitor::Result ReadStateCreator::processNodeTopDown( simulation::Node* gnode)
{
    using namespace sofa::defaulttype;
    sofa::core::componentmodel::behavior::BaseMechanicalState * mstate = dynamic_cast<sofa::core::componentmodel::behavior::BaseMechanicalState *>( gnode->getMechanicalState());
    if (!mstate)   return Visitor::RESULT_CONTINUE;
    //We have a mechanical state
    addReadState(mstate, gnode);
    return simulation::Visitor::RESULT_CONTINUE;
}

void ReadStateCreator::addReadState(sofa::core::componentmodel::behavior::BaseMechanicalState *ms, simulation::Node* gnode)
{
    sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping; context->get(mapping);
    if (createInMapping || mapping== NULL)
    {
        sofa::component::misc::ReadState *rs; context->get(rs);
        if (  rs == NULL )
        {
            rs = new sofa::component::misc::ReadState(); gnode->addObject(rs);
        }

        std::ostringstream ofilename;
        ofilename << sceneName << "_" << counterReadState << "_" << ms->getName()  << "_mstate.txt" ;

        rs->f_filename.setValue(ofilename.str());  rs->f_listening.setValue(false); //Desactivated only called by extern functions
        if (init) rs->init();

        ++counterReadState;
    }
}

///if state is true, we activate all the write states present in the scene.
simulation::Visitor::Result ReadStateActivator::processNodeTopDown( simulation::Node* gnode)
{
    sofa::component::misc::ReadState *rs = gnode->get< sofa::component::misc::ReadState >();
    if (rs != NULL) { changeStateReader(rs);}

    return simulation::Visitor::RESULT_CONTINUE;
}

void ReadStateActivator::changeStateReader(sofa::component::misc::ReadState* rs)
{
    rs->reset();
    rs->f_listening.setValue(state);
}


//if state is true, we activate all the write states present in the scene. If not, we activate all the readers.
simulation::Visitor::Result ReadStateModifier::processNodeTopDown( simulation::Node* gnode)
{
    using namespace sofa::defaulttype;

    sofa::component::misc::ReadState*rs = gnode->get< sofa::component::misc::ReadState>();
    if (rs != NULL) {changeTimeReader(rs);}

    return simulation::Visitor::RESULT_CONTINUE;
}

} // namespace misc

} // namespace component

} // namespace sofa
