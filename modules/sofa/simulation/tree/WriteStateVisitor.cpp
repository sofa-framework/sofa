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
#include <sofa/simulation/tree/WriteStateVisitor.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace simulation
{

namespace tree
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

    sofa::core::objectmodel::BaseObject * mstate = gnode->getMechanicalState();
    bool found = false;
    //We have a mechanical state

#ifndef SOFA_FLOAT
    if (!found)
    {
        sofa::core::componentmodel::behavior::MechanicalState< Vec3dTypes > *ms = dynamic_cast< sofa::core::componentmodel::behavior::MechanicalState< Vec3dTypes > *>(mstate);
        if (ms!=NULL) {   addWriteState(ms, gnode);   found = true;}
    }
    if (!found)
    {
        sofa::core::componentmodel::behavior::MechanicalState< Rigid3dTypes > *ms = dynamic_cast< sofa::core::componentmodel::behavior::MechanicalState< Rigid3dTypes > *>(mstate);
        if (ms!=NULL) { addWriteState(ms, gnode);  found = true;}
    }
#endif
#ifndef SOFA_DOUBLE
    if (!found)
    {
        sofa::core::componentmodel::behavior::MechanicalState< Vec3fTypes > *ms = dynamic_cast< sofa::core::componentmodel::behavior::MechanicalState< Vec3fTypes > *>(mstate);
        if (ms != NULL) { addWriteState(ms, gnode);  found = true;}
    }
    if (!found)
    {
        sofa::core::componentmodel::behavior::MechanicalState< Rigid3fTypes > *ms = dynamic_cast< sofa::core::componentmodel::behavior::MechanicalState< Rigid3fTypes > *>(mstate);
        if (ms!=NULL) { addWriteState(ms, gnode);  found = true;}
    }
#endif


    return Visitor::RESULT_CONTINUE;
}

template< class DataTypes >
void WriteStateCreator::addWriteState(sofa::core::componentmodel::behavior::MechanicalState< DataTypes > *ms, simulation::Node* gnode)
{
    sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping;
    context->get(mapping);
    if ( mapping == NULL)
    {
        sofa::component::misc::WriteState<DataTypes> *ws;
        context->get(ws);
        if ( ws == NULL )
        {
            ws = new sofa::component::misc::WriteState<DataTypes>(); gnode->addObject(ws);
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

    sofa::core::objectmodel::BaseObject * mstate = gnode->getMechanicalState();
    bool found = false;
    //We have a mechanical state
#ifndef SOFA_FLOAT
    if (!found)
    {
        sofa::core::componentmodel::behavior::MechanicalState< Vec3dTypes > *ms = dynamic_cast< sofa::core::componentmodel::behavior::MechanicalState< Vec3dTypes > *>(mstate);
        if (ms != NULL) {  addReadState(ms, gnode);  found = true;}
    }
    if (!found)
    {
        sofa::core::componentmodel::behavior::MechanicalState< Rigid3dTypes > *ms = dynamic_cast< sofa::core::componentmodel::behavior::MechanicalState< Rigid3dTypes > *>(mstate);
        if (ms != NULL) { addReadState(ms, gnode);  found = true;}
    }
#endif
#ifndef SOFA_DOUBLE
    if (!found)
    {
        sofa::core::componentmodel::behavior::MechanicalState< Vec3fTypes > *ms = dynamic_cast< sofa::core::componentmodel::behavior::MechanicalState< Vec3fTypes > *>(mstate);
        if (ms!=NULL) {  addReadState(ms, gnode);  found = true;}
    }
    if (!found)
    {
        sofa::core::componentmodel::behavior::MechanicalState< Rigid3fTypes > *ms = dynamic_cast< sofa::core::componentmodel::behavior::MechanicalState< Rigid3fTypes > *>(mstate);
        if (ms!=NULL) {  addReadState(ms, gnode);  found = true;}
    }
#endif

    return Visitor::RESULT_CONTINUE;
}



template< class DataTypes >
void ReadStateCreator::addReadState(sofa::core::componentmodel::behavior::MechanicalState< DataTypes > *ms, simulation::Node* gnode)
{

    sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping; context->get(mapping);
    if (mapping== NULL)
    {
        sofa::component::misc::ReadState<DataTypes> *rs; context->get(rs);
        if (  rs == NULL )
        {
            rs = new sofa::component::misc::ReadState <DataTypes>(); gnode->addObject(rs);
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

    bool reader_done = false;
#ifndef SOFA_FLOAT
    if (!reader_done)
    {
        sofa::component::misc::ReadState< Vec3dTypes > *rsd = gnode->get< sofa::component::misc::ReadState< Vec3dTypes > >();
        if (rsd != NULL) { rsd->reset();  rsd->f_listening.setValue(!state); reader_done = true;}
    }
    if (!reader_done)
    {
        sofa::component::misc::ReadState< Rigid3dTypes > *rsr = gnode->get< sofa::component::misc::ReadState< Rigid3dTypes > >();
        if (rsr != NULL) { rsr->reset();  rsr->f_listening.setValue(!state); reader_done = true;}
    }
#endif
#ifndef SOFA_DOUBLE
    if (!reader_done)
    {
        sofa::component::misc::ReadState< Vec3fTypes > *rsf = gnode->get< sofa::component::misc::ReadState< Vec3fTypes > >();
        if (rsf != NULL) { rsf->reset();  rsf->f_listening.setValue(!state); reader_done = true;}
    }
    if (!reader_done)
    {
        sofa::component::misc::ReadState< Rigid3fTypes > *rsr = gnode->get< sofa::component::misc::ReadState< Rigid3fTypes > >();
        if (rsr != NULL) { rsr->reset();  rsr->f_listening.setValue(!state); reader_done = true;}
    }
#endif
    bool writer_done = false;
#ifndef SOFA_FLOAT
    if (!writer_done)
    {
        sofa::component::misc::WriteState< Vec3dTypes > *wsd = gnode->get< sofa::component::misc::WriteState< Vec3dTypes > >();
        if (wsd != NULL) { changeStateWriter(wsd); writer_done = true;}
    }
    if (!writer_done)
    {
        sofa::component::misc::WriteState< Rigid3dTypes > *wsr = gnode->get< sofa::component::misc::WriteState< Rigid3dTypes > >();
        if (wsr != NULL) { changeStateWriter(wsr); writer_done = true;}
    }
#endif
#ifndef SOFA_DOUBLE
    if (!writer_done)
    {
        sofa::component::misc::WriteState< Vec3fTypes > *wsf = gnode->get< sofa::component::misc::WriteState< Vec3fTypes > >();
        if (wsf != NULL) { changeStateWriter(wsf); writer_done = true;}
    }
    if (!writer_done)
    {
        sofa::component::misc::WriteState< Rigid3fTypes > *wsr = gnode->get< sofa::component::misc::WriteState< Rigid3fTypes > >();
        if (wsr != NULL) { changeStateWriter(wsr); writer_done = true;}
    }
#endif

    return Visitor::RESULT_CONTINUE;
}

template< class DataTypes >
void WriteStateActivator::changeStateWriter(sofa::component::misc::WriteState< DataTypes > *ws)
{
    if (!state) ws->reset();
    ws->f_listening.setValue(state);
}


//if state is true, we activate all the write states present in the scene. If not, we activate all the readers.
Visitor::Result ReadStateModifier::processNodeTopDown( simulation::Node* gnode)
{
    using namespace sofa::defaulttype;

#ifndef SOFA_FLOAT
    sofa::component::misc::ReadState< Vec3dTypes > *rsdf = gnode->get< sofa::component::misc::ReadState< Vec3dTypes > >();
    if (rsdf != NULL) {changeTimeReader(rsdf);}
    sofa::component::misc::ReadState< Rigid3dTypes > *rsrf = gnode->get< sofa::component::misc::ReadState< Rigid3dTypes > >();
    if (rsrf != NULL) { changeTimeReader(rsrf);}
#endif
#ifndef SOFA_DOUBLE
    sofa::component::misc::ReadState< Rigid3fTypes > *rsrd = gnode->get< sofa::component::misc::ReadState< Rigid3fTypes > >();
    if (rsrd != NULL) { changeTimeReader(rsrd);}
    sofa::component::misc::ReadState< Vec3fTypes > *rsfd = gnode->get< sofa::component::misc::ReadState< Vec3fTypes > >();
    if (rsfd != NULL) {changeTimeReader(rsfd); }
#endif

    return Visitor::RESULT_CONTINUE;
}
}
}
}

