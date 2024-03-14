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
#include <sofa/simulation/mechanicalvisitor/MechanicalIntegrateConstraintVisitor.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
namespace sofa::simulation::mechanicalvisitor
{

MechanicalIntegrateConstraintsVisitor::MechanicalIntegrateConstraintsVisitor(
        const core::ConstraintParams* cparams,
        double pf, double vf,
        sofa::core::ConstMultiVecDerivId correction,
        sofa::core::MultiVecDerivId dx,
        sofa::core::MultiVecCoordId x,
        sofa::core::MultiVecDerivId v)
    :BaseMechanicalVisitor(cparams)
    ,cparams(cparams)
    ,positionFactor(pf)
    ,velocityFactor(vf)
    ,correctionId(correction)
    ,dxId(dx)
    ,xId(x)
    ,vId(v)
    ,offset(0)
{}

MechanicalIntegrateConstraintsVisitor::Result MechanicalIntegrateConstraintsVisitor::fwdMechanicalState(simulation::Node* /*node*/, core::behavior::BaseMechanicalState* ms)
{
    if (positionFactor != 0)
    {
        //x = x_free + correction * positionFactor;
        ms->vOp(params, xId.getId(ms), cparams->x().getId(ms), correctionId.getId(ms), positionFactor);
    }

    if (velocityFactor != 0)
    {
        //v = v_free + correction * velocityFactor;
        ms->vOp(params, vId.getId(ms), cparams->v().getId(ms), correctionId.getId(ms), velocityFactor);
    }

    const double correctionFactor = cparams->constOrder() == sofa::core::ConstraintOrder::VEL ? velocityFactor : positionFactor;

    //dx *= correctionFactor;
    ms->vOp(params,dxId.getId(ms),core::VecDerivId::null(), correctionId.getId(ms), correctionFactor);

    return RESULT_CONTINUE;
}

} // namespace sofa::simulation::mechanicalvisitor

