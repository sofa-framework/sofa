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
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/Node.h>

#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace simulation
{

Visitor::Result UpdateContextVisitor::processNodeTopDown(simulation::Node* node)
{

    if (!startingNode) startingNode = node;
    if (node != startingNode)
        node->updateContext();
    return RESULT_CONTINUE;
}

Visitor::Result UpdateSimulationContextVisitor::processNodeTopDown(simulation::Node* node)
{
    if (!startingNode) startingNode = node;
    if (node != startingNode)
        node->updateSimulationContext();
    return RESULT_CONTINUE;
}


UpdateVisualContextVisitor::UpdateVisualContextVisitor(const sofa::core::visual::VisualParams* vparams)
	: UpdateContextVisitor(vparams)
{

}


Visitor::Result UpdateVisualContextVisitor::processNodeTopDown(simulation::Node* node)
{

    if (!startingNode) startingNode = node;
    if (node != startingNode)
        node->updateVisualContext();
    return RESULT_CONTINUE;
}


} // namespace simulation

} // namespace sofa

