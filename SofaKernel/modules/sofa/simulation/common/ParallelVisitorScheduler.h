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
#ifndef SOFA_SIMULATION_COMMON_PARALLELACTIONSCHEDULER_H
#define SOFA_SIMULATION_COMMON_PARALLELACTIONSCHEDULER_H


#include <sofa/simulation/common/VisitorScheduler.h>
#include <sofa/simulation/common/Node.h>

namespace sofa
{

namespace simulation
{


/// Specialized VisitorScheduler for parallel implementations.
class SOFA_SIMULATION_COMMON_API ParallelVisitorScheduler : public simulation::VisitorScheduler
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
