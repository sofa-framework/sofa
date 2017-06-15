/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_VISITORSCHEDULER_H
#define SOFA_SIMULATION_VISITORSCHEDULER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/simulation/simulationcore.h>
#include "ClassSystem.h"

namespace sofa
{

namespace simulation
{


class Visitor;

/// Abstract class defining custom schedule of action execution through the graph.
class SOFA_SIMULATION_CORE_API VisitorScheduler : public virtual core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(VisitorScheduler, core::objectmodel::BaseObject);

    virtual void executeVisitor(simulation::Node* node, Visitor* act) = 0;

    /// Specify whether this scheduler is multi-threaded.
    virtual bool isMultiThreaded() const { return false; }

protected:

    VisitorScheduler() {}

    virtual ~VisitorScheduler() {}

    /// Execute the given action recursively
    void doExecuteVisitor(simulation::Node* node, Visitor* act);
	
private:
	VisitorScheduler(const VisitorScheduler& n) ;
	VisitorScheduler& operator=(const VisitorScheduler& n) ;
	
	
};

} // namespace simulation

} // namespace sofa

#endif
