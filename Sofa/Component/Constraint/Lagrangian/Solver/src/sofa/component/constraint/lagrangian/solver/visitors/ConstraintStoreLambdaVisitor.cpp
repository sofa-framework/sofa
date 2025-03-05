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
#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/component/constraint/lagrangian/solver/visitors/ConstraintStoreLambdaVisitor.h>
#include <sofa/core/ConstraintParams.h>

namespace sofa::component::constraint::lagrangian::solver
{

ConstraintStoreLambdaVisitor::ConstraintStoreLambdaVisitor(const sofa::core::ConstraintParams* cParams, const sofa::linearalgebra::BaseVector* lambda)
:simulation::BaseMechanicalVisitor(cParams)
,m_cParams(cParams)
,m_lambda(lambda)
{
}

simulation::Visitor::Result ConstraintStoreLambdaVisitor::fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet)
{
    if (core::behavior::BaseConstraint *c = dynamic_cast<core::behavior::BaseConstraint*>(cSet) )
    {
        const ctime_t t0 = begin(node, c);
        c->storeLambda(m_cParams, m_cParams->lambda(), m_lambda);
        end(node, c, t0);
    }
    return RESULT_CONTINUE;
}

void ConstraintStoreLambdaVisitor::bwdMechanicalMapping(simulation::Node* node, core::BaseMapping* map)
{
    SOFA_UNUSED(node);

    sofa::core::MechanicalParams mparams(*m_cParams);
    mparams.setDx(m_cParams->dx());
    mparams.setF(m_cParams->lambda());
    map->applyJT(&mparams, m_cParams->lambda(), m_cParams->lambda());
}

bool ConstraintStoreLambdaVisitor::stopAtMechanicalMapping(simulation::Node* node, core::BaseMapping* map)
{
    SOFA_UNUSED(node);
    SOFA_UNUSED(map);

    return false;
}

} // namespace sofa::component::constraint::lagrangian::solver
