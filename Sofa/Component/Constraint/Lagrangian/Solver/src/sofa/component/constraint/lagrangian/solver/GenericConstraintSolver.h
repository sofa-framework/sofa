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
#include <sofa/component/constraint/lagrangian/solver/GenericConstraintProblem.h>

#include <sofa/component/constraint/lagrangian/solver/ConstraintSolverImpl.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>
#include <sofa/core/behavior/BaseLagrangianConstraint.h>
#include <sofa/helper/map.h>

#include <sofa/simulation/CpuTask.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/component/constraint/lagrangian/solver/visitors/MechanicalGetConstraintResolutionVisitor.h>
#include <sofa/helper/SelectableItem.h>


namespace sofa::component::constraint::lagrangian::solver
{

class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API GenericConstraintSolver : public ConstraintSolverImpl
{
    typedef sofa::core::MultiVecId MultiVecId;
    friend GenericConstraintProblem;
public:
    SOFA_CLASS(GenericConstraintSolver, ConstraintSolverImpl);
protected:
    GenericConstraintSolver();
    ~GenericConstraintSolver() override;
public:
    void init() override;

    void cleanup() override;

    bool prepareStates(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;
    bool buildSystem(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;
    void rebuildSystem(const SReal massFactor, const SReal forceFactor) override;
    bool solveSystem(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;
    bool applyCorrection(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;
    void computeResidual(const core::ExecParams* /*params*/) override;
    ConstraintProblem* getConstraintProblem() override;
    void lockConstraintProblem(sofa::core::objectmodel::BaseObject* from, ConstraintProblem* p1, ConstraintProblem* p2 = nullptr) override;


    Data<int> d_maxIt; ///< maximal number of iterations of iterative algorithm
    Data<SReal> d_tolerance; ///< residual error threshold for termination of the Gauss-Seidel algorithm
    Data<SReal> d_sor; ///< Successive Over Relaxation parameter (0-2)
    Data< SReal > d_regularizationTerm; ///< add regularization*Id to W when solving for constraints
    Data<bool> d_scaleTolerance; ///< Scale the error tolerance with the number of constraints
    Data<bool> d_allVerified; ///< All constraints must be verified (each constraint's error < tolerance)

    Data<bool> d_computeGraphs; ///< Compute graphs of errors and forces during resolution
    Data<std::map < std::string, sofa::type::vector<SReal> > > d_graphErrors; ///< Sum of the constraints' errors at each iteration
    Data<std::map < std::string, sofa::type::vector<SReal> > > d_graphConstraints; ///< Graph of each constraint's error at the end of the resolution
    Data<std::map < std::string, sofa::type::vector<SReal> > > d_graphForces; ///< Graph of each constraint's force at each step of the resolution
    Data<std::map < std::string, sofa::type::vector<SReal> > > d_graphViolations; ///< Graph of each constraint's violation at each step of the resolution

    Data<int> d_currentNumConstraints; ///< OUTPUT: current number of constraints
    Data<int> d_currentNumConstraintGroups; ///< OUTPUT: current number of constraints
    Data<int> d_currentIterations; ///< OUTPUT: current number of constraint groups
    Data<SReal> d_currentError; ///< OUTPUT: current error
    Data<type::vector< SReal >> d_constraintForces; ///< OUTPUT: constraint forces (stored only if computeConstraintForces=True)
    Data<bool> d_computeConstraintForces; ///< The indices of the constraintForces to store in the constraintForce data field.

    sofa::core::MultiVecDerivId getLambda() const override;
    sofa::core::MultiVecDerivId getDx() const override;

protected:

    void clearConstraintProblemLocks();

    static constexpr auto CP_BUFFER_SIZE = 10;
    sofa::type::fixed_array<GenericConstraintProblem * , CP_BUFFER_SIZE> m_cpBuffer;
    sofa::type::fixed_array<bool, CP_BUFFER_SIZE> m_cpIsLocked;
    GenericConstraintProblem *current_cp, *last_cp;

    sofa::core::MultiVecDerivId m_lambdaId;
    sofa::core::MultiVecDerivId m_dxId;

    virtual void initializeConstraintProblems();

    /*****
     *
     * @brief This internal method is used to build the system. It should use the list of constraint correction (l_constraintCorrections) to build the full constraint problem.
     *
     * @param cParams: Container providing quick access to all data related to the mechanics (position, velocity etc..) for all mstate
     * @param problem: constraint problem containing data structures used for solving the constraint
     *                 problem: the constraint matrix, the unknown vector, the free violation etc...
     *                 The goal of this method is to fill parts of this structure to be then used to
     *                 find the unknown vector.
     * @param numConstraints: number of atomic constraint
     *
     */
    virtual void doBuildSystem( const core::ConstraintParams *cParams, GenericConstraintProblem * problem ,unsigned int numConstraints) = 0;

    /*****
     *
     * @brief This internal method is used to solve the constraint problem. It essentially uses the constraint problem structures.
     *
     * @param problem: constraint problem containing data structures used for solving the constraint
     *                 problem: the constraint matrix, the unknown vector, the free violation etc...
     *                 The goal of this method is to use the problem structures to compute the final solution.
     * @param timeout: timeout to use this solving method in a haptic thread. If the timeout is reached then the solving must stops.
     *
     */
    virtual void doSolve( GenericConstraintProblem * problem, SReal timeout = 0.0) = 0;


    static void addRegularization(linearalgebra::BaseMatrix& W, const SReal regularization);


private:

    sofa::type::vector<core::behavior::BaseConstraintCorrection*> filteredConstraintCorrections() const;

    void computeAndApplyMotionCorrection(const core::ConstraintParams* cParams, GenericConstraintSolver::MultiVecId res1, GenericConstraintSolver::MultiVecId res2) const;
    void applyMotionCorrection(
        const core::ConstraintParams* cParams,
        MultiVecId res1, MultiVecId res2,
        core::behavior::BaseConstraintCorrection* constraintCorrection) const;

    // Accumulate the lambda values projected in the motion space in the states
    // f += J^T * lambda
    void storeConstraintLambdas(const core::ConstraintParams* cParams);

};

} //namespace sofa::component::constraint::lagrangian::solver
