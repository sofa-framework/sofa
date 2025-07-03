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

class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_CORRECTION_API GenericConstraintCorrection
    : public core::behavior::BaseConstraintCorrection
{
public:
    SOFA_CLASS(GenericConstraintCorrection, core::behavior::BaseConstraintCorrection);

    void init() override;
    void cleanup() override;
    void addConstraintSolver(core::behavior::ConstraintSolver *s) override;
    void removeConstraintSolver(core::behavior::ConstraintSolver *s) override;

    /**
     * \brief \copybrief BaseConstraintCorrection::computeMotionCorrectionFromLambda
     *
     * \details Calls the linear solver to perform the computations:
     * f = J^T * lambda
     * dx = A^-1 * f
     *
     * f is implicitly stored in cparams->lambda()
     */
    void computeMotionCorrectionFromLambda(
        const core::ConstraintParams* cparams,
        core::MultiVecDerivId dx,
        const linearalgebra::BaseVector * lambda) override;

    /**
     * \brief \copybrief BaseConstraintCorrection::addComplianceInConstraintSpace
     *
     * \details Calls the linear solver to perform the computation W += J A^-1 J^T
     */
    void addComplianceInConstraintSpace(const core::ConstraintParams *cparams, linearalgebra::BaseMatrix* W) override;

    void getComplianceMatrix(linearalgebra::BaseMatrix* ) const override;

    /**
     * Compute:
     * x = x_free + correction * positionFactor
     * v = v_free + correction * velocityFactor
     * dx *= correctionFactor
     *
     * x_free and v_free correspond to the position and velocity of the free motion. Both vectors
     * are referred in \p cparams.
     *
     * positionFactor and velocityFactor are factors provided by the ODE solver.
     *
     * correctionFactor is either positionFactor or velocityFactor depending on
     * cparams->constOrder()
     */
    void applyMotionCorrection(
        const core::ConstraintParams *cparams,
        core::MultiVecCoordId x,
        core::MultiVecDerivId v,
        core::MultiVecDerivId dx,
        core::ConstMultiVecDerivId correction) override;

    /**
     * Compute:
     * x = x_free + correction * positionFactor
     * dx *= correctionFactor
     *
     * x_free corresponds to the position of the free motion. x_free is referred in \p cparams.
     *
     * positionFactor is a factor provided by the ODE solver.
     */
    void applyPositionCorrection(const core::ConstraintParams *cparams, core::MultiVecCoordId x, core::MultiVecDerivId dx, core::ConstMultiVecDerivId correction) override;

    /**
     * Compute:
     * v = v_free + correction * velocityFactor
     * dx *= correctionFactor
     *
     * v_free corresponds to the velocity of the free motion. v_free is referred in \p cparams.
     *
     * velocityFactor is a factor provided by the ODE solver.
     */
    void applyVelocityCorrection(const core::ConstraintParams *cparams, core::MultiVecDerivId v, core::MultiVecDerivId dv, core::ConstMultiVecDerivId correction) override;

    void applyPredictiveConstraintForce(const core::ConstraintParams *cparams, core::MultiVecDerivId f, const linearalgebra::BaseVector *lambda) override;

    void rebuildSystem(SReal massFactor, SReal forceFactor) override;

    void applyContactForce(const linearalgebra::BaseVector *f) override;

    void resetContactForce() override;

    void computeResidual(const core::ExecParams* params, linearalgebra::BaseVector *lambda) override;

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
};

} //namespace sofa::component::constraint::lagrangian::correction
