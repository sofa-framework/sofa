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
#ifndef SOFA_SIMULATION_CORE_PARALLELACTIONSCHEDULER_H
#define SOFA_SIMULATION_CORE_PARALLELACTIONSCHEDULER_H


#include <sofa/simulation/VisitorScheduler.h>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace simulation
{


/// Specialized VisitorScheduler for parallel implementations.
class SOFA_SIMULATION_CORE_API ParallelVisitorScheduler : public simulation::VisitorScheduler
{
public:
    ParallelVisitorScheduler(bool propagate=false);

    /// Specify whether this scheduler is multi-threaded.
    virtual bool isMultiThreaded() const { return true; }

    virtual void executeVisitor(Node* node, Visitor* action);

protected:
    bool propagate;

    void recursiveClone(Node* node);

    virtual ParallelVisitorScheduler* clone() = 0;
    virtual void executeParallelVisitor(Node* node, Visitor* action) = 0;
};

} // namespace simulation

} // namespace sofa

#endif
