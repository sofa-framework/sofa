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
// C++ Interface: PropagateEventVisitor
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_SIMULATION_TREE_PROPAGATEEVENTACTION_H
#define SOFA_SIMULATION_TREE_PROPAGATEEVENTACTION_H

#include <sofa/simulation/tree/Visitor.h>
#include <sofa/core/objectmodel/Event.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

/**
Visitor used to propagate an event in the the data structure.
Propagation is done top-down until the event is handled.

	@author The SOFA team </www.sofa-framework.org>
*/
class PropagateEventVisitor : public sofa::simulation::tree::Visitor
{
public:
    PropagateEventVisitor(sofa::core::objectmodel::Event* e);

    ~PropagateEventVisitor();

    Visitor::Result processNodeTopDown(GNode* node);
    void processObject(GNode*, core::objectmodel::BaseObject* obj);

protected:
    sofa::core::objectmodel::Event* m_event;
};

}

}

}

#endif
