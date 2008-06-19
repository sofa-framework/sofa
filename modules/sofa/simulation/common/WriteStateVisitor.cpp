/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
//
// C++ Implementation: WriteStateVisitor
//
// Description:
//
//
// Author: Francois Faure, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/simulation/common/WriteStateVisitor.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace simulation
{


WriteStateVisitor::WriteStateVisitor( std::ostream& out )
    : m_out(out)
{}

WriteStateVisitor::~WriteStateVisitor()
{}

Visitor::Result WriteStateVisitor::processNodeTopDown( simulation::Node* gnode )
{
    for( simulation::Node::ObjectIterator i=gnode->object.begin(), iend=gnode->object.end(); i!=iend; i++ )
    {
        (*i)->writeState(m_out);
    }
    return Visitor::RESULT_CONTINUE;
}


//Create a Write State component each time a mechanical state is found
Visitor::Result WriteStateCreator::processNodeTopDown( simulation::Node* gnode)
{
    using namespace sofa::defaulttype;

    sofa::core::componentmodel::behavior::BaseMechanicalState * mstate = dynamic_cast<sofa::core::componentmodel::behavior::BaseMechanicalState *>( gnode->getMechanicalState());
    if (!mstate)   return Visitor::RESULT_CONTINUE;
    //We have a mechanical state
    addWriteState(mstate, gnode);
    return Visitor::RESULT_CONTINUE;
}



void WriteStateCreator::addWriteState(sofa::core::componentmodel::behavior::BaseMechanicalState *ms, simulation::Node* gnode)
{
    sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping;
    context->get(mapping);
    if ( mapping == NULL)
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


//Create a Read State component each time a mechanical state is found
Visitor::Result ReadStateCreator::processNodeTopDown( simulation::Node* gnode)
{
    using namespace sofa::defaulttype;
    sofa::core::componentmodel::behavior::BaseMechanicalState * mstate = dynamic_cast<sofa::core::componentmodel::behavior::BaseMechanicalState *>( gnode->getMechanicalState());
    if (!mstate)   return Visitor::RESULT_CONTINUE;
    //We have a mechanical state
    addReadState(mstate, gnode);
    return Visitor::RESULT_CONTINUE;
}



void ReadStateCreator::addReadState(sofa::core::componentmodel::behavior::BaseMechanicalState *ms, simulation::Node* gnode)
{

    sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping; context->get(mapping);
    if (mapping== NULL)
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

//if state is true, we activate all the write states present in the scene.
Visitor::Result WriteStateActivator::processNodeTopDown( simulation::Node* gnode)
{
    using namespace sofa::defaulttype;
    sofa::component::misc::ReadState *rs = gnode->get< sofa::component::misc::ReadState >();
    if (rs != NULL) { rs->reset();  rs->f_listening.setValue(!state);}

    sofa::component::misc::WriteState *ws = gnode->get< sofa::component::misc::WriteState >();
    if (ws != NULL) { changeStateWriter(ws);}
    return Visitor::RESULT_CONTINUE;
}

void WriteStateActivator::changeStateWriter(sofa::component::misc::WriteState*ws)
{
    if (!state) ws->reset();
    ws->f_listening.setValue(state);
}


//if state is true, we activate all the write states present in the scene. If not, we activate all the readers.
Visitor::Result ReadStateModifier::processNodeTopDown( simulation::Node* gnode)
{
    using namespace sofa::defaulttype;

    sofa::component::misc::ReadState*rs = gnode->get< sofa::component::misc::ReadState>();
    if (rs != NULL) {changeTimeReader(rs);}

    return Visitor::RESULT_CONTINUE;
}
}
}

