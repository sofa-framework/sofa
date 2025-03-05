/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/playback/ReadState.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/simulation/Node.h>

namespace sofa::component::playback
{

using namespace defaulttype;

void registerReadState(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Read State vectors from file at each timestep.")
        .add< ReadState >());
}

ReadStateCreator::ReadStateCreator(const core::ExecParams* params)
    : Visitor(params)
    , sceneName("")
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    , extension(".txt.gz")
#else
    , extension(".txt")
#endif
    , createInMapping(false)
    , init(true)
    , counterReadState(0)
{
}

ReadStateCreator::ReadStateCreator(const std::string &n, bool _createInMapping, const core::ExecParams* params, bool i, int c)
    : Visitor(params)
    , sceneName(n)
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    , extension(".txt.gz")
#else
    , extension(".txt")
#endif
    , createInMapping(_createInMapping)
    , init(i)
    , counterReadState(c)
{
}

//Create a Read State component each time a mechanical state is found
simulation::Visitor::Result ReadStateCreator::processNodeTopDown( simulation::Node* gnode)
{
    sofa::core::behavior::BaseMechanicalState * mstate=gnode->mechanicalState;
    if (!mstate)   return Visitor::RESULT_CONTINUE;
    core::behavior::OdeSolver *isSimulated;
    mstate->getContext()->get(isSimulated);
    if (!isSimulated) return simulation::Visitor::RESULT_CONTINUE;

    //We have a mechanical state
    addReadState(mstate, gnode);
    return simulation::Visitor::RESULT_CONTINUE;
}

void ReadStateCreator::addReadState(sofa::core::behavior::BaseMechanicalState *ms, simulation::Node* gnode)
{
    const sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping; context->get(mapping);
    if (createInMapping || mapping== nullptr)
    {
        sofa::component::playback::ReadState::SPtr rs;
        context->get(rs, this->subsetsToManage, core::objectmodel::BaseContext::Local);
        if (rs == nullptr)
        {
            rs = sofa::core::objectmodel::New<ReadState>();
            gnode->addObject(rs);
            for (const auto& subset : this->subsetsToManage)
            {
                rs->addTag(subset);
            }
        }

        std::ostringstream ofilename;
        ofilename << sceneName << "_" << counterReadState << "_" << ms->getName()  << "_mstate" << extension ;

        rs->d_filename.setValue(ofilename.str());  rs->f_listening.setValue(false); //Deactivated only called by extern functions
        if (init) rs->init();

        ++counterReadState;
    }
}

///if state is true, we activate all the write states present in the scene.
simulation::Visitor::Result ReadStateActivator::processNodeTopDown( simulation::Node* gnode)
{
    sofa::component::playback::ReadState *rs = gnode->get< sofa::component::playback::ReadState >(this->subsetsToManage);
    if (rs != nullptr) { changeStateReader(rs);}

    return simulation::Visitor::RESULT_CONTINUE;
}

void ReadStateActivator::changeStateReader(sofa::component::playback::ReadState* rs)
{
    rs->reset();
    rs->f_listening.setValue(state);
}


//if state is true, we activate all the write states present in the scene. If not, we activate all the readers.
simulation::Visitor::Result ReadStateModifier::processNodeTopDown( simulation::Node* gnode)
{
    using namespace sofa::defaulttype;

    sofa::component::playback::ReadState*rs = gnode->get< sofa::component::playback::ReadState>(this->subsetsToManage);
    if (rs != nullptr) {changeTimeReader(rs);}

    return simulation::Visitor::RESULT_CONTINUE;
}

} // namespace sofa::component::playback
