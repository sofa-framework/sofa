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

    /**
    * !!! WARNING since v25.12 !!!
    *
    * The template method pattern has been applied to this part of the API.
    * This method calls the newly introduced method "doAddComplianceInConstraintSpace" internally,
    * which is the method to override from now on.
    *
    **/
    /// @name Compliance Matrix API
    /// @{
    virtual void addComplianceInConstraintSpace(const ConstraintParams *, linearalgebra::BaseMatrix* W) final;
    virtual void doAddComplianceInConstraintSpace(const ConstraintParams *, linearalgebra::BaseMatrix* W) = 0;

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doGetComplianceMatrix" internally,
     * which is the method to override from now on.
     *
     **/
    /// Fill the matrix m with the full Compliance Matrix
    virtual void getComplianceMatrix(linearalgebra::BaseMatrix* m) final;
    virtual void doGetComplianceMatrix(linearalgebra::BaseMatrix* m) const = 0;

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doGetComplianceWithConstraintMerge" internally,
     * which is the method to override from now on.
     *
     **/
    /// For multigrid approach => constraints are merged
    virtual void getComplianceWithConstraintMerge(linearalgebra::BaseMatrix* /*Wmerged*/, std::vector<int> & /*constraint_merge*/) final;
    virtual void doGetComplianceWithConstraintMerge(linearalgebra::BaseMatrix* /*Wmerged*/, std::vector<int> & /*constraint_merge*/);

    /// @}

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doAddConstraintSolver" internally,
     * which is the method to override from now on.
     *
     **/
    /// Keeps track of the constraint solver
    ///
    /// @param s is the constraint solver
    virtual void addConstraintSolver(ConstraintSolver *s) final;
    virtual void doAddConstraintSolver(ConstraintSolver *s) = 0;


    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doRemoveConstraintSolver" internally,
     * which is the method to override from now on.
     *
     **/
    /// Remove reference to constraint solver
    ///
    /// @param s is the constraint solver
    virtual void removeConstraintSolver(ConstraintSolver *s) final;
    virtual void doRemoveConstraintSolver(ConstraintSolver *s) = 0;


    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doComputeMotionCorrectionFromLambda" internally,
     * which is the method to override from now on.
     *
     **/
    /// Compute the corrective motion from the constraint space lambda
    ///
    /// @param cparams the ConstraintParams relative to the constraint solver
    /// @param dx the VecId where to store the corrective motion
    /// @param lambda is the constraint space force vector
    virtual void computeMotionCorrectionFromLambda(const core::ConstraintParams* cparams, core::MultiVecDerivId dx, const linearalgebra::BaseVector * lambda) final;
    virtual void doComputeMotionCorrectionFromLambda(const core::ConstraintParams* cparams, core::MultiVecDerivId dx, const linearalgebra::BaseVector * lambda) = 0;


    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doApplyMotionCorrection" internally,
     * which is the method to override from now on.
     *
     **/
    /// Compute motion correction from the constraint resolution (LCP) calculated force
    ///
    /// @param cparams
    /// @param x is the position result VecId
    /// @param v is the velocity result VecId
    /// @param dx if the corrective motion result VecId
    virtual void applyMotionCorrection(const ConstraintParams * cparams, MultiVecCoordId x, MultiVecDerivId v, MultiVecDerivId dx, ConstMultiVecDerivId correction) final;
    virtual void doApplyMotionCorrection(const ConstraintParams * cparams, MultiVecCoordId x, MultiVecDerivId v, MultiVecDerivId dx, ConstMultiVecDerivId correction) = 0;

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doApplyPositionCorrection" internally,
     * which is the method to override from now on.
     *
     **/
    /// Compute position correction from the constraint resolution (LCP) calculated force
    ///
    /// @param cparams
    /// @param x is the position result VecId
    /// @param v is the velocity result VecId
    /// @param dx if the corrective position result VecId
    virtual void applyPositionCorrection(const ConstraintParams * cparams, MultiVecCoordId x, MultiVecDerivId dx, ConstMultiVecDerivId correction) final;
    virtual void doApplyPositionCorrection(const ConstraintParams * cparams, MultiVecCoordId x, MultiVecDerivId dx, ConstMultiVecDerivId correction) = 0;


    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doApplyVelocityCorrection" internally,
     * which is the method to override from now on.
     *
     **/
    /// Compute velocity correction from the constraint resolution (LCP) calculated force
    ///
    /// @param cparams
    /// @param x is the position result VecId
    /// @param v is the velocity result VecId
    /// @param dv if the corrective velocity result VecId
    /// @param correction is the corrective motion computed from the constraint lambda
    virtual void applyVelocityCorrection(const ConstraintParams * cparams, MultiVecDerivId v, MultiVecDerivId dv, ConstMultiVecDerivId correction) final;
    virtual void doApplyVelocityCorrection(const ConstraintParams * cparams, MultiVecDerivId v, MultiVecDerivId dv, ConstMultiVecDerivId correction) = 0;

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doApplyPredictiveConstraintForce" internally,
     * which is the method to override from now on.
     *
     **/
    /// Apply predictive constraint force
    ///
    /// @param cparams
    /// @param f is the motion space force vector
    /// @param lambda is the constraint space force vector
    virtual void applyPredictiveConstraintForce(const ConstraintParams * cparams, MultiVecDerivId f, const linearalgebra::BaseVector * lambda) final;
    virtual void doApplyPredictiveConstraintForce(const ConstraintParams * cparams, MultiVecDerivId f, const linearalgebra::BaseVector * lambda) = 0;

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doRebuildSystem" internally,
     * which is the method to override from now on.
     *
     **/
    /// Rebuild the system using a mass and force factor
    /// Experimental API used to investigate convergence issues.
    virtual void rebuildSystem(SReal /*massFactor*/, SReal /*forceFactor*/) final;
    virtual void doRebuildSystem(SReal /*massFactor*/, SReal /*forceFactor*/);

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doComputeResidual" internally,
     * which is the method to override from now on.
     *
     **/
    /// Compute the residual in the newton iterations due to the constraints forces
    /// i.e. compute Vecid::force() += J^t lambda
    /// the result is accumulated in Vecid::force()
    virtual void computeResidual(const core::ExecParams* /*params*/, linearalgebra::BaseVector * /*lambda*/) ;
    virtual void doComputeResidual(const core::ExecParams* /*params*/, linearalgebra::BaseVector * /*lambda*/) ;

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doApplyContactForce" internally,
     * which is the method to override from now on.
     *
     **/
    virtual void applyContactForce(const linearalgebra::BaseVector *f) final;
    virtual void doApplyContactForce(const linearalgebra::BaseVector *f) = 0;

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doResetContactForce" internally,
     * which is the method to override from now on.
     *
     **/
    virtual void resetContactForce() final;
    virtual void doResetContactForce() = 0;

    /// @name Unbuilt constraint system during resolution
    /// @{
    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doHasConstraintNumber" internally,
     * which is the method to override from now on.
     *
     **/
    virtual bool hasConstraintNumber(int /*index*/);
    virtual bool doHasConstraintNumber(int /*index*/);

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doResetForUnbuiltResolution" internally,
     * which is the method to override from now on.
     *
     **/
    virtual void resetForUnbuiltResolution(SReal* /*f*/, std::list<unsigned int>& /*renumbering*/) final;
    virtual void doResetForUnbuiltResolution(SReal* /*f*/, std::list<unsigned int>& /*renumbering*/);

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doAddConstraintDisplacement" internally,
     * which is the method to override from now on.
     *
     **/
    virtual void addConstraintDisplacement(SReal* /*d*/, int /*begin*/, int /*end*/) final;
    virtual void doAddConstraintDisplacement(SReal* /*d*/, int /*begin*/, int /*end*/);

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doSetConstraintDForce" internally,
     * which is the method to override from now on.
     *
     **/
    virtual void setConstraintDForce(SReal* /*df*/, int /*begin*/, int /*end*/, bool /*update*/) final;	  // f += df
    virtual void doSetConstraintDForce(SReal* /*df*/, int /*begin*/, int /*end*/, bool /*update*/);	  // f += df

    /**
     * !!! WARNING since v25.12 !!!
     *
     * The template method pattern has been applied to this part of the API.
     * This method calls the newly introduced method "doGetBlockDiagonalCompliance" internally,
     * which is the method to override from now on.
     *
     **/
    virtual void getBlockDiagonalCompliance(linearalgebra::BaseMatrix* /*W*/, int /*begin*/,int /*end*/) final;
    virtual void doGetBlockDiagonalCompliance(linearalgebra::BaseMatrix* /*W*/, int /*begin*/,int /*end*/);
    /// @}

protected:
    BaseConstraintCorrection();
    ~BaseConstraintCorrection() override;

private:
    BaseConstraintCorrection(const BaseConstraintCorrection& n) = delete ;
    BaseConstraintCorrection& operator=(const BaseConstraintCorrection& n) = delete ;

};

} // namespace sofa::core::behavior
