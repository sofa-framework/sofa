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
#include <sofa/component/playback/WriteState.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/BaseMapping.h>

namespace sofa::component::playback
{

using namespace defaulttype;


int WriteStateClass = core::RegisterObject("Write State vectors to file at each timestep")
        .add< WriteState >();

WriteStateCreator::WriteStateCreator(const core::ExecParams* params)
    :simulation::Visitor(params)
    , sceneName("")
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    , extension(".txt.gz")
#else
    , extension(".txt")
#endif
    , recordX(true)
    , recordV(true)
    , createInMapping(false)
    , counterWriteState(0)
{
}

WriteStateCreator::WriteStateCreator(const core::ExecParams* params, const std::string &n, bool _recordX, bool _recordV, bool _recordF, bool _createInMapping, int c)
    :simulation::Visitor(params)
    , sceneName(n)
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    , extension(".txt.gz")
#else
    , extension(".txt")
#endif
    , recordX(_recordX)
    , recordV(_recordV)
    , recordF(_recordF)
    , createInMapping(_createInMapping)
    , counterWriteState(c)
{
}


//Create a Write State component each time a mechanical state is found
simulation::Visitor::Result WriteStateCreator::processNodeTopDown( simulation::Node* gnode)
{
    sofa::core::behavior::BaseMechanicalState * mstate=gnode->mechanicalState;
    if (!mstate)   return simulation::Visitor::RESULT_CONTINUE;
    core::behavior::OdeSolver *isSimulated;
    mstate->getContext()->get(isSimulated);
    if (!isSimulated) return simulation::Visitor::RESULT_CONTINUE;

    //We have a mechanical state
    addWriteState(mstate, gnode);
    return simulation::Visitor::RESULT_CONTINUE;
}


void WriteStateCreator::addWriteState(sofa::core::behavior::BaseMechanicalState *ms, simulation::Node* gnode)
{
    const sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping;
    context->get(mapping);
    if ( createInMapping || mapping == nullptr)
    {
        sofa::component::playback::WriteState::SPtr ws;
        context->get(ws, this->subsetsToManage, core::objectmodel::BaseContext::Local);
        if ( ws == nullptr )
        {
            ws = sofa::core::objectmodel::New<WriteState>();
            gnode->addObject(ws);
            ws->d_writeX.setValue(recordX);
            ws->d_writeV.setValue(recordV);
            ws->d_writeF.setValue(recordF);
            for (core::objectmodel::TagSet::iterator it=this->subsetsToManage.begin(); it != this->subsetsToManage.end(); ++it)
                ws->addTag(*it);

        }
        std::ostringstream ofilename;
        ofilename << sceneName << "_" << counterWriteState << "_" << ms->getName()  << "_mstate" << extension ;

        ws->d_filename.setValue(ofilename.str());
        if (!m_times.empty())
            ws->d_time.setValue(m_times);

        ws->init();
        ws->f_listening.setValue(true);  //Activated at init
        
        ++counterWriteState;
    }
}



//if state is true, we activate all the write states present in the scene.
simulation::Visitor::Result WriteStateActivator::processNodeTopDown( simulation::Node* gnode)
{
    sofa::component::playback::WriteState *ws = gnode->get< sofa::component::playback::WriteState >(this->subsetsToManage);
    if (ws != nullptr) { changeStateWriter(ws);}
    return simulation::Visitor::RESULT_CONTINUE;
}

void WriteStateActivator::changeStateWriter(sofa::component::playback::WriteState*ws)
{
    if (!state) ws->reset();
    ws->f_listening.setValue(state);
}

} // namespace sofa::component::playback
