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
#include <sofa/component/linearsystem/visitors/DispatchFromGlobalVectorToLocalVectorVisitor.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa::component::linearsystem
{

DispatchFromGlobalVectorToLocalVectorVisitor::DispatchFromGlobalVectorToLocalVectorVisitor(const core::ExecParams* params,
    const MappingGraph& mappingGraph, sofa::core::MultiVecId dst, linearalgebra::BaseVector* globalVector)
    : BaseMechanicalVisitor(params)
    , m_dst(dst)
    , m_globalVector(globalVector)
    , m_mappingGraph(mappingGraph)
{}

simulation::Visitor::Result DispatchFromGlobalVectorToLocalVectorVisitor::fwdMechanicalState(simulation::Node* node,
                                                                                           core::behavior::BaseMechanicalState* mm)
{
    SOFA_UNUSED(node);

    if (mm)
    {
        auto pos = m_mappingGraph.getPositionInGlobalMatrix(mm);
        mm->copyFromBaseVector(m_dst.getId(mm), m_globalVector, pos[0]);
    }
    return RESULT_CONTINUE;
}

} //namespace sofa::component::linearsystem
