/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
//
// C++ Implementation: PropagateEventVisitor
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/simulation/common/PropagateEventVisitor.h>

namespace sofa
{

namespace simulation
{


PropagateEventVisitor::PropagateEventVisitor(const core::ExecParams* params, sofa::core::objectmodel::Event* e)
    : sofa::simulation::Visitor(params)
    , m_event(e)
{}


PropagateEventVisitor::~PropagateEventVisitor()
{}

Visitor::Result PropagateEventVisitor::processNodeTopDown(simulation::Node* node)
{
    if( (node->m_mask & (1 << m_event->getEventTypeIndex())) != 0 ){
        for_each(this, node, node->object, &PropagateEventVisitor::processObject) ;

        if( m_event->isHandled() )
            return Visitor::RESULT_PRUNE;
        else
            return Visitor::RESULT_CONTINUE;
    }
    return Visitor::RESULT_CONTINUE;
}

void PropagateEventVisitor::processObject(simulation::Node*, core::objectmodel::BaseObject* obj)
{
    //if( obj->f_listening.getValue()==true )
    //std::cout << "processObject: " << obj->getName() << std::endl;
    //std::cout << "mask: " << obj->m_mask << " event " << m_event->getEventTypeIndex() << std::endl ;
    //std::cout <<  (obj->m_mask & (1 << m_event->getEventTypeIndex())) << std::endl ;
    if( (obj->m_mask & (1 << m_event->getEventTypeIndex())) != 0){
        obj->handleEvent( m_event );
    }/*else{
        std::cout << "Event not propagated to : " << m_event->getClassName() << " to " << obj->getName() << std::endl ;
        std::cout << "Why " << m_event->getEventTypeIndex() << std::endl ;
    }*/
}


}

}

