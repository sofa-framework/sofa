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
#pragma once
#include <sofa/component/constraint/lagrangian/correction/config.h>

#include <sofa/core/behavior/BaseConstraintCorrection.h>

namespace sofa::component::constraint::lagrangian::correction
{

class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API GenericConstraintCorrection : public core::behavior::BaseConstraintCorrection
{
public:
    SOFA_CLASS(GenericConstraintCorrection, core::behavior::BaseConstraintCorrection);

    void bwdInit() override;
    void cleanup() override;

    SingleLink<GenericConstraintCorrection, sofa::core::behavior::LinearSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_linearSolver; ///< Link towards the linear solver used to compute the compliance matrix, requiring the inverse of the linear system matrix
    SingleLink<GenericConstraintCorrection, sofa::core::behavior::OdeSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_ODESolver; ///< Link towards the ODE solver used to recover the integration factors
    Data< SReal > d_complianceFactor; ///< Factor applied to the position factor and velocity factor used to calculate compliance matrix
    Data< SReal > d_regularizationTerm; ///< add regularizationTerm*Id to W when solving for constraints

protected:
    GenericConstraintCorrection();
    ~GenericConstraintCorrection() override;

    std::list<core::behavior::ConstraintSolver*> constraintsolvers;

    void applyMotionCorrection(const core::ConstraintParams* cparams, core::MultiVecCoordId xId, core::MultiVecDerivId vId, core::MultiVecDerivId dxId,
                               core::ConstMultiVecDerivId correction, SReal positionFactor, SReal velocityFactor);

    void doAddConstraintSolver(core::behavior::ConstraintSolver *s) override;

    void doRemoveConstraintSolver(core::behavior::ConstraintSolver *s) override;

    void doComputeMotionCorrectionFromLambda(const core::ConstraintParams* cparams, core::MultiVecDerivId dx, const linearalgebra::BaseVector * lambda) override;

    void doAddComplianceInConstraintSpace(const core::ConstraintParams *cparams, linearalgebra::BaseMatrix* W) override;

    void doGetComplianceMatrix(linearalgebra::BaseMatrix* ) const override;

    void doApplyMotionCorrection(const core::ConstraintParams *cparams, core::MultiVecCoordId x, core::MultiVecDerivId v, core::MultiVecDerivId dx, core::ConstMultiVecDerivId correction) override;

    void doApplyPositionCorrection(const core::ConstraintParams *cparams, core::MultiVecCoordId x, core::MultiVecDerivId dx, core::ConstMultiVecDerivId correction) override;

    void doApplyVelocityCorrection(const core::ConstraintParams *cparams, core::MultiVecDerivId v, core::MultiVecDerivId dv, core::ConstMultiVecDerivId correction) override;

    void doApplyPredictiveConstraintForce(const core::ConstraintParams *cparams, core::MultiVecDerivId f, const linearalgebra::BaseVector *lambda) override;

    void doRebuildSystem(SReal massFactor, SReal forceFactor) override;

    void doApplyContactForce(const linearalgebra::BaseVector *f) override;

    void doResetContactForce() override;

    void doComputeResidual(const core::ExecParams* params, linearalgebra::BaseVector *lambda) override;

};

} //namespace sofa::component::constraint::lagrangian::correction
