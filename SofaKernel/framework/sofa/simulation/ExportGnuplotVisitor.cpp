/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sofa/simulation/ExportGnuplotVisitor.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <iostream>

namespace sofa
{

namespace simulation
{

simulation::Visitor::Result InitGnuplotVisitor::processNodeTopDown(simulation::Node* node)
{
    if (node->interactionForceField.getSize() != 0)
    {
        int size = node->interactionForceField.getSize();
        for(int i = 0; i < size; i++)
        {
            if (node->interactionForceField.getValue()[i] )
            {
                node->interactionForceField.getValue()[i]->initGnuplot(gnuplotDirectory);
            }
        }
    }

    if (node->mechanicalState)
    {
        node->mechanicalState->initGnuplot(gnuplotDirectory);
    }
    if (node->mass)
    {
        node->mass->initGnuplot(gnuplotDirectory);
    }
    return RESULT_CONTINUE;
}

ExportGnuplotVisitor::ExportGnuplotVisitor(const core::ExecParams* params, SReal time)
    : Visitor(params), m_time(time)
{

}

simulation::Visitor::Result ExportGnuplotVisitor::processNodeTopDown(simulation::Node* node)
{
    if (node->interactionForceField.getSize() != 0)
    {
        int size = node->interactionForceField.getSize();
        for(int i = 0; i < size; i++)
        {
            if (node->interactionForceField.getValue()[i] )
            {
                node->interactionForceField.getValue()[i]->exportGnuplot(m_time);
            }
        }
    }

    if (node->mechanicalState)
    {
        node->mechanicalState->exportGnuplot(m_time);
    }
    if (node->mass)
    {
        node->mass->exportGnuplot(core::MechanicalParams::defaultInstance(), m_time);
    }
    return RESULT_CONTINUE;
}


} // namespace simulation

} // namespace sofa

