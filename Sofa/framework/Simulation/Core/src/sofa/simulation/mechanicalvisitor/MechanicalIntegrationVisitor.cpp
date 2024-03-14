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

#include <sofa/simulation/mechanicalvisitor/MechanicalIntegrationVisitor.h>

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalBeginIntegrationVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalProjectPositionAndVelocityVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalEndIntegrationVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalAccumulateMatrixDeriv.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalBuildConstraintMatrix.h>

namespace sofa::simulation::mechanicalvisitor
{

Visitor::Result MechanicalIntegrationVisitor::fwdOdeSolver(simulation::Node* node, core::behavior::OdeSolver* obj)
{
    SReal nextTime = node->getTime() + dt;
    MechanicalBeginIntegrationVisitor beginVisitor( this->params, dt );
    node->execute(&beginVisitor);

    sofa::core::MechanicalParams mparams(*this->params);
    mparams.setDt(dt);

    core::ConstraintParams cparams;
    {
        unsigned int constraintId=0;
        MechanicalBuildConstraintMatrix buildConstraintMatrix(&cparams, core::MatrixDerivId::constraintJacobian(), constraintId );
        buildConstraintMatrix.execute(node);
    }

    {
        MechanicalAccumulateMatrixDeriv accumulateMatrixDeriv(&cparams, core::MatrixDerivId::constraintJacobian());
        accumulateMatrixDeriv.execute(node);
    }

    obj->solve(params, dt);

    MechanicalProjectPositionAndVelocityVisitor(&mparams, nextTime,core::VecCoordId::position(),core::VecDerivId::velocity()
    ).execute( node );

    MechanicalPropagateOnlyPositionAndVelocityVisitor(&mparams, nextTime,core::VecCoordId::position(),core::VecDerivId::velocity()).execute( node );

    MechanicalEndIntegrationVisitor endVisitor( this->params, dt );
    node->execute(&endVisitor);

    return RESULT_PRUNE;
}

Visitor::Result MechanicalIntegrationVisitor::fwdInteractionForceField(simulation::Node* /*node*/, core::behavior::BaseInteractionForceField* obj)
{
    const core::MultiVecDerivId   ffId      = core::VecDerivId::externalForce();
    core::MechanicalParams m_mparams(*this->params);
    m_mparams.setDt(this->dt);

    obj->addForce(&m_mparams, ffId);
    return RESULT_CONTINUE;
}

}
