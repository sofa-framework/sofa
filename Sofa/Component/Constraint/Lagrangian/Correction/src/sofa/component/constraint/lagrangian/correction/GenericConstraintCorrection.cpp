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
#define SOFA_COMPONENT_CONSTRAINT_GENERICCONSTRAINTCORRECTION_CPP

#include <sofa/component/constraint/lagrangian/correction/GenericConstraintCorrection.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalIntegrateConstraintVisitor.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/ConstraintParams.h>

using sofa::core::execparams::defaultInstance; 

namespace sofa::component::constraint::lagrangian::correction 
{
using sofa::simulation::mechanicalvisitor::MechanicalIntegrateConstraintsVisitor;
using sofa::type::vector;
using sofa::core::objectmodel::BaseContext;
using sofa::core::behavior::BaseConstraintCorrection;
using sofa::core::behavior::ConstraintSolver;
using sofa::linearalgebra::BaseMatrix;
using sofa::core::ConstraintParams;
using sofa::core::MultiVecDerivId;
using sofa::core::MultiVecCoordId;
using sofa::core::ExecParams;
using sofa::linearalgebra::BaseVector;
using sofa::core::RegisterObject;
using sofa::core::ConstMultiVecDerivId;
using sofa::core::VecDerivId;
using sofa::core::VecCoordId;

GenericConstraintCorrection::GenericConstraintCorrection()
: l_linearSolver(initLink("linearSolver", "Link towards the linear solver used to compute the compliance matrix, requiring the inverse of the linear system matrix"))
, l_ODESolver(initLink("ODESolver", "Link towards the ODE solver used to recover the integration factors"))
, d_complianceFactor(initData(&d_complianceFactor, 1.0_sreal, "complianceFactor", "Factor applied to the position factor and velocity factor used to calculate compliance matrix"))
{
}

GenericConstraintCorrection::~GenericConstraintCorrection() {}

void GenericConstraintCorrection::bwdInit()
{
    const BaseContext* context = this->getContext();

    // Find linear solver
    if (l_linearSolver.empty())
    {
        msg_info() << "Link \"linearSolver\" to the desired linear solver should be set to ensure right behavior." << msgendl
                   << "First LinearSolver found in current context will be used.";
        l_linearSolver.set( context->get<sofa::core::behavior::LinearSolver>(BaseContext::Local) );
    }

    if (l_linearSolver.get() == nullptr)
    {
        msg_error() << "No LinearSolver component found at path: " << l_linearSolver.getLinkedPath() << ", nor in current context: " << context->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    else
    {
        if (l_linearSolver.get()->getTemplateName() == "GraphScattered")
        {
            msg_error() << "Can not use the solver " << l_linearSolver.get()->getName() << " because it is templated on GraphScatteredType";
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
        else
        {
            msg_info() << "LinearSolver path used: '" << l_linearSolver.getLinkedPath() << "'";
        }
    }

    // Find ODE solver
    if (l_ODESolver.empty())
    {
        msg_info() << "Link \"ODESolver\" to the desired ODE solver should be set to ensure right behavior." << msgendl
                   << "First ODESolver found in current context will be used.";
        l_ODESolver.set( context->get<sofa::core::behavior::OdeSolver>(BaseContext::Local) );
        if (l_ODESolver.get() == nullptr)
        {
            l_ODESolver.set( context->get<sofa::core::behavior::OdeSolver>(BaseContext::SearchRoot) );
        }
    }

    if (l_ODESolver.get() == nullptr)
    {
        msg_error() << "No ODESolver component found at path: " << l_ODESolver.getLinkedPath() << ", nor in current context: " << context->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    else
    {
        msg_info() << "ODESolver path used: '" << l_ODESolver.getLinkedPath() << "'";
    }

    sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void GenericConstraintCorrection::cleanup()
{
    while(!constraintsolvers.empty())
    {
        constraintsolvers.back()->removeConstraintCorrection(this);
        constraintsolvers.pop_back();
    }
    BaseConstraintCorrection::cleanup();
}

void GenericConstraintCorrection::addConstraintSolver(ConstraintSolver *s)
{
    constraintsolvers.push_back(s);
}

void GenericConstraintCorrection::removeConstraintSolver(ConstraintSolver *s)
{
    constraintsolvers.remove(s);
}

void GenericConstraintCorrection::rebuildSystem(SReal massFactor, SReal forceFactor)
{
    l_linearSolver.get()->rebuildSystem(massFactor, forceFactor);
}

void GenericConstraintCorrection::addComplianceInConstraintSpace(const ConstraintParams *cparams, BaseMatrix* W)
{
    if (!l_ODESolver.get()) return;
    const SReal complianceFactor = d_complianceFactor.getValue();

    // use the OdeSolver to get the position integration factor
    SReal factor = 1.0;

    switch (cparams->constOrder())
    {
        case sofa::core::ConstraintOrder::POS_AND_VEL :
        case sofa::core::ConstraintOrder::POS :
            factor = l_ODESolver.get()->getPositionIntegrationFactor();
            break;

        case sofa::core::ConstraintOrder::ACC :
        case sofa::core::ConstraintOrder::VEL :
            factor = l_ODESolver.get()->getVelocityIntegrationFactor();
            break;

        default :
            break;
    }

    factor *= complianceFactor;
    // use the Linear solver to compute J*inv(M)*Jt, where M is the mechanical linear system matrix
    l_linearSolver.get()->buildComplianceMatrix(cparams, W, factor);
}

void GenericConstraintCorrection::computeMotionCorrectionFromLambda(const ConstraintParams* cparams, MultiVecDerivId dx, const linearalgebra::BaseVector * lambda)
{
    l_linearSolver.get()->applyConstraintForce(cparams, dx, lambda);
}

void GenericConstraintCorrection::applyMotionCorrection(const ConstraintParams* cparams,
                                                        MultiVecCoordId xId,
                                                        MultiVecDerivId vId, 
                                                        MultiVecDerivId dxId,
                                                        ConstMultiVecDerivId correction,
                                                        SReal positionFactor,
                                                        SReal velocityFactor)
{
    MechanicalIntegrateConstraintsVisitor v(cparams, positionFactor, velocityFactor, correction, dxId, xId, vId);
    l_linearSolver.get()->getContext()->executeVisitor(&v);
}

void GenericConstraintCorrection::applyMotionCorrection(const ConstraintParams * cparams,
                                                        MultiVecCoordId xId,
                                                        MultiVecDerivId vId,
                                                        MultiVecDerivId dxId,
                                                        ConstMultiVecDerivId correction)
{
    if (!l_ODESolver.get()) return;
    const SReal complianceFactor = d_complianceFactor.getValue();

    const SReal positionFactor = l_ODESolver.get()->getPositionIntegrationFactor() * complianceFactor;
    const SReal velocityFactor = l_ODESolver.get()->getVelocityIntegrationFactor() * complianceFactor;

    applyMotionCorrection(cparams, xId, vId, dxId, correction, positionFactor, velocityFactor);
}

void GenericConstraintCorrection::applyPositionCorrection(const ConstraintParams * cparams,
                                                          MultiVecCoordId xId,
                                                          MultiVecDerivId dxId,
                                                          ConstMultiVecDerivId correctionId)
{
    if (!l_ODESolver.get()) return;

    const SReal complianceFactor = d_complianceFactor.getValue();
    const SReal positionFactor = l_ODESolver.get()->getPositionIntegrationFactor() * complianceFactor;

    applyMotionCorrection(cparams, xId, VecDerivId::null(), dxId, correctionId, positionFactor, 0);
}

void GenericConstraintCorrection::applyVelocityCorrection(const ConstraintParams * cparams,
                                                          MultiVecDerivId vId,
                                                          MultiVecDerivId dvId,
                                                          ConstMultiVecDerivId correctionId)
{
    if (!l_ODESolver.get()) return;

    const SReal velocityFactor = l_ODESolver.get()->getVelocityIntegrationFactor();

    applyMotionCorrection(cparams, VecCoordId::null(), vId, dvId, correctionId, 0, velocityFactor);
}

void GenericConstraintCorrection::applyContactForce(const BaseVector *f)
{
    if (!l_ODESolver.get()) return;

    ConstraintParams cparams(*sofa::core::execparams::defaultInstance());


    computeMotionCorrectionFromLambda(&cparams, cparams.dx(), f);
    applyMotionCorrection(&cparams, VecCoordId::position(), VecDerivId::velocity(), cparams.dx(), cparams.lambda());
}

void GenericConstraintCorrection::computeResidual(const ExecParams* params, linearalgebra::BaseVector *lambda)
{
    l_linearSolver.get()->computeResidual(params, lambda);
}


void GenericConstraintCorrection::getComplianceMatrix(linearalgebra::BaseMatrix* Minv) const
{
    if (!l_ODESolver.get())
        return;

    const ConstraintParams cparams(*sofa::core::execparams::defaultInstance());
    const_cast<GenericConstraintCorrection*>(this)->addComplianceInConstraintSpace(&cparams, Minv);
}

void GenericConstraintCorrection::applyPredictiveConstraintForce(const ConstraintParams * cparams,
                                                                 MultiVecDerivId f,
                                                                 const BaseVector * lambda)
{
    SOFA_UNUSED(cparams);
    SOFA_UNUSED(f);
    SOFA_UNUSED(lambda);
}

void GenericConstraintCorrection::resetContactForce(){}

int GenericConstraintCorrectionClass = RegisterObject("")
    .add< GenericConstraintCorrection >();

} //namespace sofa::component::constraint::lagrangian::correction
