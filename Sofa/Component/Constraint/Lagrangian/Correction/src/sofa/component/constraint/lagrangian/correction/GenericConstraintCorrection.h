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
    void addConstraintSolver(core::behavior::ConstraintSolver *s) override;
    void removeConstraintSolver(core::behavior::ConstraintSolver *s) override;

    void computeMotionCorrectionFromLambda(const core::ConstraintParams* cparams, core::MultiVecDerivId dx, const linearalgebra::BaseVector * lambda) override;

    void addComplianceInConstraintSpace(const core::ConstraintParams *cparams, linearalgebra::BaseMatrix* W) override;

    void getComplianceMatrix(linearalgebra::BaseMatrix* ) const override;

    void applyMotionCorrection(const core::ConstraintParams *cparams, core::MultiVecCoordId x, core::MultiVecDerivId v, core::MultiVecDerivId dx, core::ConstMultiVecDerivId correction) override;

    void applyPositionCorrection(const core::ConstraintParams *cparams, core::MultiVecCoordId x, core::MultiVecDerivId dx, core::ConstMultiVecDerivId correction) override;

    void applyVelocityCorrection(const core::ConstraintParams *cparams, core::MultiVecDerivId v, core::MultiVecDerivId dv, core::ConstMultiVecDerivId correction) override;

    void applyPredictiveConstraintForce(const core::ConstraintParams *cparams, core::MultiVecDerivId f, const linearalgebra::BaseVector *lambda) override;

    void rebuildSystem(SReal massFactor, SReal forceFactor) override;

    void applyContactForce(const linearalgebra::BaseVector *f) override;

    void resetContactForce() override;

    void computeResidual(const core::ExecParams* params, linearalgebra::BaseVector *lambda) override;

    SingleLink<GenericConstraintCorrection, sofa::core::behavior::LinearSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_linearSolver; ///< Link towards the linear solver used to compute the compliance matrix, requiring the inverse of the linear system matrix
    SingleLink<GenericConstraintCorrection, sofa::core::behavior::OdeSolver, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_ODESolver; ///< Link towards the ODE solver used to recover the integration factors
    Data< SReal > d_complianceFactor; ///< Factor applied to the position factor and velocity factor used to calculate compliance matrix.

    SOFA_ATTRIBUTE_DISABLED__CONSTRAINTCORRECTION_EXPLICITLINK()
    core::objectmodel::lifecycle::RemovedData d_linearSolversName {this, "v22.12", "v23.06", "solverName", "replace solverName with an explicit data link named \"linearSolver\" (PR #3152)"};

    SOFA_ATTRIBUTE_DISABLED__CONSTRAINTCORRECTION_EXPLICITLINK()
    core::objectmodel::lifecycle::RemovedData d_ODESolverName {this, "v22.12", "v23.06", "ODESolverName", "replace sODESolverName with an explicit data link named \"ODESolver\" (PR #3152)"};

protected:
    GenericConstraintCorrection();
    ~GenericConstraintCorrection() override;

    std::list<core::behavior::ConstraintSolver*> constraintsolvers;

    void applyMotionCorrection(const core::ConstraintParams* cparams, core::MultiVecCoordId xId, core::MultiVecDerivId vId, core::MultiVecDerivId dxId,
                               core::ConstMultiVecDerivId correction, SReal positionFactor, SReal velocityFactor);
};

} //namespace sofa::component::constraint::lagrangian::correction
