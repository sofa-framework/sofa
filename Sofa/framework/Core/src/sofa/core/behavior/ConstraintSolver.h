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
#include <sofa/core/behavior/BaseConstraintSet.h>

namespace sofa::core::behavior
{

class BaseConstraintCorrection;

/**
 *  \brief Component responsible for the expression and solution of system of equations related to constraints.
 The main method is solveConstraint(const ConstraintParams *, MultiVecId , MultiVecId );
 The default implementation successively calls: prepareStates, buildSystem, solveSystem, applyCorrection.
 The parameters are defined in class ConstraintParams.
 *
 */
class SOFA_CORE_API ConstraintSolver : public virtual objectmodel::BaseObject
{
public:

    SOFA_ABSTRACT_CLASS(ConstraintSolver, objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(ConstraintSolver)

protected:

    ConstraintSolver();

    ~ConstraintSolver() override;

private:
    ConstraintSolver(const ConstraintSolver& n) = delete;
    ConstraintSolver& operator=(const ConstraintSolver& n) = delete;


public:
    /**
     * Launch the sequence of operations in order to solve the constraints
     */
    virtual void solveConstraint(const ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null());

    /**
     * Do the precomputation: compute free state, or propagate the states to the mapped mechanical states, where the constraint can be expressed
     */
    virtual bool prepareStates(const ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null())=0;

    /**
     * Create the system corresponding to the constraints
     */
    virtual bool buildSystem(const ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null())=0;

    /**
     * Rebuild the system using a mass and force factor.
     * Experimental API used to investigate convergence issues.
     */
    virtual void rebuildSystem(SReal /*massfactor*/, SReal /*forceFactor*/){}

    /**
     * Use the system previously built and solve it with the appropriate algorithm
     */
    virtual bool solveSystem(const ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null())=0;

    /**
     * Correct the Mechanical State with the solution found
     */
    virtual bool applyCorrection(const ConstraintParams *, MultiVecId res1, MultiVecId res2=MultiVecId::null())=0;


    /// Compute the residual in the newton iterations due to the constraints forces
    /// i.e. compute Vecid::force() += J^t lambda
    /// the result is accumulated in Vecid::force()
    virtual void computeResidual(const core::ExecParams* /*params*/)
    {
        dmsg_error() << "ComputeResidual is not implemented in " << this->getName() ;
    }

    /// @name Resolution DOFs vectors API
    /// @{
    virtual MultiVecDerivId getLambda() const
    {
        return MultiVecDerivId(VecDerivId::externalForce());
    }

    virtual MultiVecDerivId getDx() const
    {
        return MultiVecDerivId(VecDerivId::dx());
    }
    /// @}

    /// Remove reference to ConstraintCorrection
    ///
    /// @param c is the ConstraintCorrection
    virtual void removeConstraintCorrection(BaseConstraintCorrection *s) = 0;

    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

protected:

    virtual void postBuildSystem(const ConstraintParams* constraint_params) { SOFA_UNUSED(constraint_params); }
    virtual void postSolveSystem(const ConstraintParams* constraint_params) { SOFA_UNUSED(constraint_params); }

    bool prepareStatesTask(const ConstraintParams*, MultiVecId res1, MultiVecId res2);
    bool buildSystemTask(const ConstraintParams *, MultiVecId res1, MultiVecId res2);
    bool solveSystemTask(const ConstraintParams *, MultiVecId res1, MultiVecId res2);
    bool applyCorrectionTask(const ConstraintParams *, MultiVecId res1, MultiVecId res2);

};

} // namespace sofa::core::behavior
