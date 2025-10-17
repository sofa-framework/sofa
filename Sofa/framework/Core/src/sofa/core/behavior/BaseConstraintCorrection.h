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

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/core/ConstraintOrder.h>

namespace sofa::core::behavior
{

class ConstraintSolver;

/**
 *  \brief Component computing constraint forces within a simulated body using the compliance method.
 */
class SOFA_CORE_API BaseConstraintCorrection : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseConstraintCorrection, objectmodel::BaseObject);

    virtual bool isActive() { return this->getContext()->isActive(); }

    /// @name Compliance Matrix API
    /// @{

    /// Compute the compliance matrix projected in the constraint space and accumulate it into \p W
    ///
    /// The computation is W += J A^-1 J^T where J is the constraint Jacobian matrix and A is the
    /// mechanical matrix
    virtual void addComplianceInConstraintSpace(const ConstraintParams *, linearalgebra::BaseMatrix* W) = 0;

    /// Fill the matrix m with the full Compliance Matrix
    virtual void getComplianceMatrix(linearalgebra::BaseMatrix* m) const = 0;

    /// For multigrid approach => constraints are merged
    virtual void getComplianceWithConstraintMerge(linearalgebra::BaseMatrix* /*Wmerged*/, std::vector<int> & /*constraint_merge*/);

    /// @}

    /// Keeps track of the constraint solver
    ///
    /// @param s is the constraint solver
    virtual void addConstraintSolver(ConstraintSolver *s) = 0;

    /// Remove reference to constraint solver
    ///
    /// @param s is the constraint solver
    virtual void removeConstraintSolver(ConstraintSolver *s) = 0;


    /// Compute the corrective motion from the constraint space lambda
    ///
    /// @param cparams the ConstraintParams relative to the constraint solver
    /// @param dx the VecId where to store the corrective motion
    /// @param lambda is the constraint space force vector
    virtual void computeMotionCorrectionFromLambda(const core::ConstraintParams* cparams, core::MultiVecDerivId dx, const linearalgebra::BaseVector * lambda) = 0;

    /// Compute motion correction from the constraint resolution (LCP) calculated force
    ///
    /// @param cparams
    /// @param x is the position result VecId
    /// @param v is the velocity result VecId
    /// @param dx if the corrective motion result VecId
    virtual void applyMotionCorrection(const ConstraintParams * cparams, MultiVecCoordId x, MultiVecDerivId v, MultiVecDerivId dx, ConstMultiVecDerivId correction) = 0;

    /// Compute position correction from the constraint resolution (LCP) calculated force
    ///
    /// @param cparams
    /// @param x is the position result VecId
    /// @param v is the velocity result VecId
    /// @param dx if the corrective position result VecId
    virtual void applyPositionCorrection(const ConstraintParams * cparams, MultiVecCoordId x, MultiVecDerivId dx, ConstMultiVecDerivId correction) = 0;

    /// Compute velocity correction from the constraint resolution (LCP) calculated force
    ///
    /// @param cparams
    /// @param x is the position result VecId
    /// @param v is the velocity result VecId
    /// @param dv if the corrective velocity result VecId
    /// @param correction is the corrective motion computed from the constraint lambda
    virtual void applyVelocityCorrection(const ConstraintParams * cparams, MultiVecDerivId v, MultiVecDerivId dv, ConstMultiVecDerivId correction) = 0;

    /// Apply predictive constraint force
    ///
    /// @param cparams
    /// @param f is the motion space force vector
    /// @param lambda is the constraint space force vector
    virtual void applyPredictiveConstraintForce(const ConstraintParams * cparams, MultiVecDerivId f, const linearalgebra::BaseVector * lambda) = 0;

    /// Rebuild the system using a mass and force factor
    /// Experimental API used to investigate convergence issues.
    SOFA_ATTRIBUTE_DEPRECATED__REBUILDSYSTEM() virtual void rebuildSystem(SReal /*massFactor*/, SReal /*forceFactor*/);

    /// Compute the residual in the newton iterations due to the constraints forces
    /// i.e. compute Vecid::force() += J^t lambda
    /// the result is accumulated in Vecid::force()
    SOFA_ATTRIBUTE_DEPRECATED__COMPUTERESIDUAL()
    virtual void computeResidual(const core::ExecParams* /*params*/, linearalgebra::BaseVector * /*lambda*/) ;

    virtual void applyContactForce(const linearalgebra::BaseVector *f) = 0;
    virtual void resetContactForce() = 0;

    /// @name Unbuilt constraint system during resolution
    /// @{

    /// Is the constraint managed by this constraint correction?
    virtual bool hasConstraintNumber(int /*index*/);
    virtual void resetForUnbuiltResolution(SReal* /*f*/, std::list<unsigned int>& /*renumbering*/);
    virtual void addConstraintDisplacement(SReal* /*d*/, int /*begin*/, int /*end*/);
    virtual void setConstraintDForce(SReal* /*df*/, int /*begin*/, int /*end*/, bool /*update*/);	  // f += df
    virtual void getBlockDiagonalCompliance(linearalgebra::BaseMatrix* /*W*/, int /*begin*/,int /*end*/);
    /// @}

protected:
    BaseConstraintCorrection();
    ~BaseConstraintCorrection() override;

    static SReal correctionFactor(const sofa::core::behavior::OdeSolver* solver, const ConstraintOrder& constraintOrder);

private:
    BaseConstraintCorrection(const BaseConstraintCorrection& n) = delete ;
    BaseConstraintCorrection& operator=(const BaseConstraintCorrection& n) = delete ;

};

} // namespace sofa::core::behavior
