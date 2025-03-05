/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/simulation/WriteStateVisitor.h>
#include <sofa/type/Vec.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/behavior/BaseMechanicalState.h>


namespace sofa::simulation
{


WriteStateVisitor::WriteStateVisitor( const sofa::core::ExecParams* params, std::ostream& out )
    : Visitor(params), m_out(out)
{}

WriteStateVisitor::~WriteStateVisitor()
{}

Visitor::Result WriteStateVisitor::processNodeTopDown( simulation::Node* gnode )
{
    if (gnode->mechanicalState != nullptr)
    {
        gnode->mechanicalState->writeState(m_out);
    }
    return Visitor::RESULT_CONTINUE;
}

} // namespace sofa::simulation



