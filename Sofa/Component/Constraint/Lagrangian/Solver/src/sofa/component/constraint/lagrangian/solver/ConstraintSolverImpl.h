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
#include <sofa/component/constraint/lagrangian/solver/config.h>

#include <sofa/core/behavior/ConstraintSolver.h>

#include <sofa/component/constraint/lagrangian/solver/visitors/MechanicalGetConstraintViolationVisitor.h>

#include <sofa/linearalgebra/FullMatrix.h>

#include <sofa/core/ConstraintParams.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>

namespace sofa::component::constraint::lagrangian::solver
{


class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API ConstraintProblem
{
public:
    // The compliance matrix projected in the constraint space
    // If J is the jacobian matrix of the constraints and A is the mechanical matrix of the system,
    // then W = J A^{-1} J^T
    sofa::linearalgebra::LPtrFullMatrix<SReal> W;

    // The constraint values of the "free motion" state
    sofa::linearalgebra::FullVector<SReal> dFree;

    // The lambda values from the Lagrange multipliers
    sofa::linearalgebra::FullVector<SReal> f;

    ConstraintProblem();
    virtual ~ConstraintProblem();

    SReal tolerance;
    int maxIterations;

    virtual void clear(int nbConstraints);

    // Returns the number of scalar constraints, or equivalently the number of Lagrange multipliers
    int getDimension() const { return dimension; }
    void setDimension(int dim) { dimension = dim; }

    SReal** getW() { return W.lptr(); }
    SReal* getDfree() { return dFree.ptr(); }
    SReal* getF() { return f.ptr(); }

    virtual void solveTimed(SReal tolerance, int maxIt, SReal timeout) = 0;

    unsigned getProblemId() const;

protected:
    int dimension;
    unsigned problemId;
};


class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API ConstraintSolverImpl : public sofa::core::behavior::ConstraintSolver
{
public:
    SOFA_ABSTRACT_CLASS(ConstraintSolverImpl, sofa::core::behavior::ConstraintSolver)

    ConstraintSolverImpl();
    ~ConstraintSolverImpl() override;

    void init() override;
    void cleanup() override;

    virtual ConstraintProblem* getConstraintProblem() = 0;

    /// Do not use the following LCPs until the next call to this function.
    /// This is used to prevent concurrent access to the LCP when using a LCPForceFeedback through an haptic thread.
    virtual void lockConstraintProblem(sofa::core::objectmodel::BaseObject* from, ConstraintProblem* p1, ConstraintProblem* p2=nullptr) = 0;

    void removeConstraintCorrection(core::behavior::BaseConstraintCorrection *s) override;

    MultiLink< ConstraintSolverImpl,
        core::behavior::BaseConstraintCorrection,
        BaseLink::FLAG_STOREPATH> l_constraintCorrections;

protected:

    void postBuildSystem(const core::ConstraintParams* cParams) override;
    void postSolveSystem(const core::ConstraintParams* cParams) override;

    void clearConstraintCorrections();


    /// Calls the method resetConstraint on all the mechanical states and BaseConstraintSet
    /// In the case of a MechanicalObject, it clears the constraint jacobian matrix
    void resetConstraints(const core::ConstraintParams* cParams);

    /// Call the method buildConstraintMatrix on all the BaseConstraintSet
    void buildLocalConstraintMatrix(const core::ConstraintParams* cparams, unsigned int &constraintId);

    /// Calls the method applyJT on all the mappings to project the mapped
    /// constraint matrices on the main constraint matrix
    void accumulateMatrixDeriv(const core::ConstraintParams* cparams);

    /// Reset and build the constraint matrix, including the projection from
    /// the mapped DoFs
    /// \return The number of constraints, i.e. the size of the constraint matrix
    unsigned int buildConstraintMatrix(const core::ConstraintParams* cparams);

    void applyProjectiveConstraintOnConstraintMatrix(const core::ConstraintParams* cparams);

    void getConstraintViolation(const core::ConstraintParams* cparams, sofa::linearalgebra::BaseVector *v);

};

} //namespace sofa::component::constraint::lagrangian::solver
