/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/simulation/tree/ParallelVisitorScheduler.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/Visitor.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

ParallelVisitorScheduler::ParallelVisitorScheduler(bool propagate)
    : propagate(propagate)
{
}

void ParallelVisitorScheduler::executeVisitor(GNode* node, simulation::Visitor* action)
{
    // first make sure all child nodes have schedulers
    if (propagate)
        recursiveClone(node);
    // then execute sequentially non-threadsafe actions
    if (!action->isThreadSafe())
        doExecuteVisitor(node, action);
    else // or launch Cilk tasks for threadsafe actions
    {
        executeParallelVisitor(node, action);
    }
}

void ParallelVisitorScheduler::recursiveClone(GNode* node)
{
    if (!node->actionScheduler)
        node->addObject( this->clone() );
    for (unsigned int i=0; i<node->child.size(); i++)
        recursiveClone(node->child[i]);
}

} // namespace tree

} // namespace simulation

} // namespace sofa

