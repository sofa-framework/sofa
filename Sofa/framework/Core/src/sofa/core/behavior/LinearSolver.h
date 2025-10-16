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

#include <sofa/core/MultiVecId.h>
#include <sofa/core/behavior/BaseLinearSolver.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/core/behavior/BaseMatrixLinearSystem.h>

namespace sofa::core::behavior
{

/**
 *  \brief Abstract interface for linear system solvers
 *
 */
class SOFA_CORE_API LinearSolver : public BaseLinearSolver
{
public:
    SOFA_ABSTRACT_CLASS(LinearSolver, BaseLinearSolver);
    SOFA_BASE_CAST_IMPLEMENTATION(LinearSolver)
public:

    virtual sofa::core::behavior::BaseMatrixLinearSystem* getLinearSystem() const = 0;

    /// Indicate if the solver updates the system in parallel
    virtual bool isAsyncSolver() { return false; }

    /// Returns true if the solver supports non-symmetric systems
    virtual bool supportNonSymmetricSystem() const { return false; }

    /// Solve the system as constructed using the previous methods
    virtual void solveSystem() = 0;

    ///
    virtual void init_partial_solve();

    ///
    virtual void partial_solve(std::list<linearalgebra::BaseMatrix::Index>& /*I_last_Disp*/,
                               std::list<linearalgebra::BaseMatrix::Index>& /*I_last_Dforce*/,
                               bool /*NewIn*/);

    /// Invert the system, this method is optional because it's called when solveSystem() is called
    /// for the first time
    virtual void invertSystem() {}

    /// Multiply the inverse of the system matrix by the transpose of the given matrix J
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @param fact integrator parameter
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    virtual bool addMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact)
    {
        SOFA_UNUSED(result);
        SOFA_UNUSED(J);
        SOFA_UNUSED(fact);
        return false;
    }

    /// Build the jacobian of the constraints using a visitor
    ///
    /// @param cparams contains the MultiMatrixDerivId  which allows to retrieve the constraint jacobian to use for 
    ///        each mechanical object. 
    /// @param result the variable where the result will be added
    /// @param fact integrator parameter
    /// @param regularizationTerm term used to regularize the matrix
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    virtual bool buildComplianceMatrix(const sofa::core::ConstraintParams* cparams, linearalgebra::BaseMatrix* result, SReal fact, SReal regularizationTerm)
    {
        SOFA_UNUSED(cparams);
        SOFA_UNUSED(result);
        SOFA_UNUSED(fact);
        SOFA_UNUSED(regularizationTerm);
        msg_error() << "buildComplianceMatrix has not been implemented.";
        return false;
    }

    /// Apply the contactforce dx = Minv * J^t * f and store the result in dx VecId
    virtual void applyConstraintForce(const sofa::core::ConstraintParams* /*cparams*/,
                                      sofa::core::MultiVecDerivId /*dx*/,
                                      const linearalgebra::BaseVector* /*f*/);

    /// Compute the residual in the newton iterations due to the constraints forces
    /// i.e. compute mparams->dF() = J^t lambda
    /// the result is written in mparams->dF()
    virtual void computeResidual(const core::ExecParams* /*params*/,
                                 linearalgebra::BaseVector* /*f*/);

    /// Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply
    /// the result with the given matrix J
    ///
    /// This method can compute the Schur complement of the constrained system:
    /// W = H A^{-1} H^T, where:
    /// - A is the mechanical matrix
    /// - H is the constraints matrix
    /// - W is the compliance matrix projected in the constraints space
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @param fact integrator parameter
    /// @return false if the solver does not support this operation, of it the system matrix is not
    /// invertible
    virtual bool addJMInvJt(linearalgebra::BaseMatrix* result, linearalgebra::BaseMatrix* J, SReal fact)
    {
        SOFA_UNUSED(result);
        SOFA_UNUSED(J);
        SOFA_UNUSED(fact);
        return false;
    }

    /// Read the Matrix solver from a file
    virtual bool readFile(std::istream& /*in*/) { return false;}

    /// Read the Matrix solver from a file
    virtual bool writeFile(std::ostream& /*out*/) {return false;}
};

} // namespace sofa::core::behavior
