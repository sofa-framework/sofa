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

Visitor::Result WriteStateVisitor::processNodeTopDown( GNode* gnode )
{
    for( GNode::ObjectIterator i=gnode->object.begin(), iend=gnode->object.end(); i!=iend; i++ )
    {
        (*i)->writeState(m_out);
    }
    return Visitor::RESULT_CONTINUE;
}


//Create a Write State component each time a mechanical state is found
Visitor::Result WriteStateCreator::processNodeTopDown( GNode* gnode)
{
    using sofa::defaulttype::Vec3fTypes;
    using sofa::defaulttype::Vec3dTypes;
    using sofa::defaulttype::RigidTypes;

    sofa::core::objectmodel::BaseObject * mstate = gnode->getMechanicalState();
    //We have a mechanical state
    if      (sofa::core::componentmodel::behavior::MechanicalState< Vec3fTypes > *ms = dynamic_cast< sofa::core::componentmodel::behavior::MechanicalState< Vec3fTypes > *>(mstate))
    {
        addWriteState(ms, gnode);
    }
    else if (sofa::core::componentmodel::behavior::MechanicalState< Vec3dTypes > *ms = dynamic_cast< sofa::core::componentmodel::behavior::MechanicalState< Vec3dTypes > *>(mstate))
    {
        addWriteState(ms, gnode);
    }
    else if (sofa::core::componentmodel::behavior::MechanicalState< RigidTypes > *ms = dynamic_cast< sofa::core::componentmodel::behavior::MechanicalState< RigidTypes > *>(mstate))
    {
        addWriteState(ms, gnode);
    }

    return Visitor::RESULT_CONTINUE;
}


template< class DataTypes >
void WriteStateCreator::addWriteState(sofa::core::componentmodel::behavior::MechanicalState< DataTypes > *ms, GNode* gnode)
{
//     if (gnode->get< sofa::core::BaseMapping >() == NULL)
    {
        if ( gnode->get< sofa::component::misc::WriteState<DataTypes> >() == NULL )
        {
            sofa::component::misc::ReadState <DataTypes> *rs = new sofa::component::misc::ReadState <DataTypes>(); gnode->addObject(rs);
            sofa::component::misc::WriteState<DataTypes> *ws = new sofa::component::misc::WriteState<DataTypes>(); gnode->addObject(ws);

            std::ostringstream ofilename;
            ofilename << sceneName << "_" << counterWriteState << "_" << ms->getName()  << "_mstate.txt" ;

            rs->f_filename.setValue(ofilename.str()); rs->init(); rs->f_listening.setValue(false); //Desactivated only called by extern functions
            ws->f_filename.setValue(ofilename.str()); ws->init(); ws->f_listening.setValue(true);  //Activated at init

        }

        ++counterWriteState;
    }
}

//if state is true, we activate all the write states present in the scene.
Visitor::Result WriteStateActivator::processNodeTopDown( GNode* gnode)
{
    using sofa::defaulttype::Vec3fTypes;
    using sofa::defaulttype::Vec3dTypes;
    using sofa::defaulttype::RigidTypes;

    bool reader_done = false;
    sofa::component::misc::ReadState< Vec3fTypes > *rsf = gnode->get< sofa::component::misc::ReadState< Vec3fTypes > >();
    if (rsf != NULL) { rsf->reset(); reader_done = true;}
    if (!reader_done)
    {
        sofa::component::misc::ReadState< Vec3dTypes > *rsd = gnode->get< sofa::component::misc::ReadState< Vec3dTypes > >();
        if (rsd != NULL) { rsd->reset(); reader_done = true;}
        if (!reader_done)
        {
            sofa::component::misc::ReadState< RigidTypes > *rsr = gnode->get< sofa::component::misc::ReadState< RigidTypes > >();
            if (rsr != NULL) { rsr->reset(); reader_done = true;}
        }
    }
    bool writer_done = false;
    sofa::component::misc::WriteState< Vec3fTypes > *wsf = gnode->get< sofa::component::misc::WriteState< Vec3fTypes > >();
    if (wsf != NULL) { changeStateWriter(wsf); writer_done = true;}
    if (!writer_done)
    {
        sofa::component::misc::WriteState< Vec3dTypes > *wsd = gnode->get< sofa::component::misc::WriteState< Vec3dTypes > >();
        if (wsd != NULL) { changeStateWriter(wsd); writer_done = true;}
        if (!writer_done)
        {
            sofa::component::misc::WriteState< RigidTypes > *wsr = gnode->get< sofa::component::misc::WriteState< RigidTypes > >();
            if (wsr != NULL) { changeStateWriter(wsr); writer_done = true;}
        }
    }

    return Visitor::RESULT_CONTINUE;
}

template< class DataTypes >
void WriteStateActivator::changeStateWriter(sofa::component::misc::WriteState< DataTypes > *ws)
{
    if (!state) ws->reset();
    ws->f_listening.setValue(state);
}


//if state is true, we activate all the write states present in the scene. If not, we activate all the readers.
Visitor::Result ReadStateModifier::processNodeTopDown( GNode* gnode)
{
    using sofa::defaulttype::Vec3fTypes;
    using sofa::defaulttype::Vec3dTypes;
    using sofa::defaulttype::RigidTypes;

    sofa::component::misc::ReadState< Vec3fTypes > *rsf = gnode->get< sofa::component::misc::ReadState< Vec3fTypes > >();
    if (rsf != NULL) {changeTimeReader(rsf); }

    sofa::component::misc::ReadState< Vec3dTypes > *rsd = gnode->get< sofa::component::misc::ReadState< Vec3dTypes > >();
    if (rsd != NULL) {changeTimeReader(rsd);}

    sofa::component::misc::ReadState< RigidTypes > *rsr = gnode->get< sofa::component::misc::ReadState< RigidTypes > >();
    if (rsr != NULL) { changeTimeReader(rsr);}

    return Visitor::RESULT_CONTINUE;
}
}
}
}

