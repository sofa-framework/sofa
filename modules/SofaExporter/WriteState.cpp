/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaExporter/WriteState.inl>
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

WriteStateCreator::WriteStateCreator(const core::ExecParams* params)
    :simulation::Visitor(params)
    , sceneName("")
#ifdef SOFA_HAVE_ZLIB
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
#ifdef SOFA_HAVE_ZLIB
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
    sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping;
    context->get(mapping);
    if ( createInMapping || mapping == NULL)
    {
        sofa::component::misc::WriteState::SPtr ws;
        context->get(ws, this->subsetsToManage, core::objectmodel::BaseContext::Local);
        if ( ws == NULL )
        {
            ws = sofa::core::objectmodel::New<WriteState>();
            gnode->addObject(ws);
            ws->f_writeX.setValue(recordX);
            ws->f_writeV.setValue(recordV);
            ws->f_writeF.setValue(recordF);
            for (core::objectmodel::TagSet::iterator it=this->subsetsToManage.begin(); it != this->subsetsToManage.end(); ++it)
                ws->addTag(*it);

        }
        std::ostringstream ofilename;
        ofilename << sceneName << "_" << counterWriteState << "_" << ms->getName()  << "_mstate" << extension ;

        ws->f_filename.setValue(ofilename.str()); ws->init(); ws->f_listening.setValue(true);  //Activated at init

        ++counterWriteState;

    }
}



//if state is true, we activate all the write states present in the scene.
simulation::Visitor::Result WriteStateActivator::processNodeTopDown( simulation::Node* gnode)
{
    sofa::component::misc::WriteState *ws = gnode->get< sofa::component::misc::WriteState >(this->subsetsToManage);
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
