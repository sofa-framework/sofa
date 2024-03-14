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
#include <numeric>
#include <sofa/simulation/mechanicalvisitor/MechanicalIdentityBlocksInJacobianVisitor.h>

#include "sofa/core/BaseMapping.h"
#include "sofa/simulation/Node.h"

namespace sofa::simulation::mechanicalvisitor
{

MechanicalIdentityBlocksInJacobianVisitor::MechanicalIdentityBlocksInJacobianVisitor(
    const sofa::core::ExecParams* params, sofa::core::MatrixDerivId id)
    : BaseMechanicalVisitor(params)
    , m_id(id)
{
}

Visitor::Result MechanicalIdentityBlocksInJacobianVisitor::fwdMechanicalMapping(simulation::Node* node,
    sofa::core::BaseMapping* map)
{
    SOFA_UNUSED(node);
    const auto parents = map->getMechFrom();

    //insert mechanical states which have children
    listParentMStates.insert(parents.begin(), parents.end());

    return Result::RESULT_CONTINUE;
}

void MechanicalIdentityBlocksInJacobianVisitor::bwdMappedMechanicalState(simulation::Node* node,
    sofa::core::behavior::BaseMechanicalState* mm)
{
    SOFA_UNUSED(node);
    if (listParentMStates.find(mm) == listParentMStates.end())
    {
        //this mechanical state does not have any children

        sofa::type::vector<unsigned int> listAffectedDoFs(mm->getSize());
        std::iota(listAffectedDoFs.begin(), listAffectedDoFs.end(), 0);

        mm->buildIdentityBlocksInJacobian(listAffectedDoFs, m_id);
    }
}

bool MechanicalIdentityBlocksInJacobianVisitor::stopAtMechanicalMapping(simulation::Node* node,
    core::BaseMapping* base_mapping)
{
    SOFA_UNUSED(node);
    SOFA_UNUSED(base_mapping);

    return false;
}
} //namespace sofa::simulation::mechanicalvisitor

