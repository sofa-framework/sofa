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
#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/helper/map.h>

#include <sofa/simulation/CpuTask.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/component/constraint/lagrangian/solver/visitors/MechanicalGetConstraintResolutionVisitor.h>

namespace sofa::component::constraint::lagrangian::solver
{

class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API GenericConstraintSolver : public ConstraintSolverImpl
{
    typedef sofa::core::MultiVecId MultiVecId;

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
    void rebuildSystem(SReal massFactor, SReal forceFactor) override;
    bool solveSystem(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;
    bool applyCorrection(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;
    void computeResidual(const core::ExecParams* /*params*/) override;
    ConstraintProblem* getConstraintProblem() override;
    void lockConstraintProblem(sofa::core::objectmodel::BaseObject* from, ConstraintProblem* p1, ConstraintProblem* p2 = nullptr) override;

    Data< sofa::helper::OptionsGroup > d_resolutionMethod; ///< Method used to solve the constraint problem, among: \"ProjectedGaussSeidel\", \"UnbuiltGaussSeidel\" or \"for NonsmoothNonlinearConjugateGradient\"

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<int> maxIt;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<SReal> tolerance;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<SReal> sor;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<bool> scaleTolerance;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<bool> allVerified;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<bool> computeGraphs;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<std::map < std::string, sofa::type::vector<SReal> > >  graphErrors;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<std::map < std::string, sofa::type::vector<SReal> > >  graphConstraints;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<std::map < std::string, sofa::type::vector<SReal> > > graphForces;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<std::map < std::string, sofa::type::vector<SReal> > > graphViolations;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<unsigned int> currentNumConstraints;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<unsigned int> currentNumConstraintGroups;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<unsigned int> currentIterations;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<SReal> currentError;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONSTRAINT_LAGRANGIAN_SOLVER()
    Data<bool> reverseAccumulateOrder;

    Data<int> d_maxIt; ///< maximal number of iterations of the Gauss-Seidel algorithm
    Data<SReal> d_tolerance; ///< residual error threshold for termination of the Gauss-Seidel algorithm
    Data<SReal> d_sor; ///< Successive Over Relaxation parameter (0-2)
    Data<bool> d_scaleTolerance; ///< Scale the error tolerance with the number of constraints
    Data<bool> d_allVerified; ///< All contraints must be verified (each constraint's error < tolerance)
    Data<int> d_newtonIterations; ///< Maximum iteration number of Newton (for the NNCG solver only)
    Data<bool> d_multithreading; ///< Compliances built concurrently
    Data<bool> d_computeGraphs; ///< Compute graphs of errors and forces during resolution
    Data<std::map < std::string, sofa::type::vector<SReal> > > d_graphErrors; ///< Sum of the constraints' errors at each iteration
    Data<std::map < std::string, sofa::type::vector<SReal> > > d_graphConstraints; ///< Graph of each constraint's error at the end of the resolution
    Data<std::map < std::string, sofa::type::vector<SReal> > > d_graphForces; ///< Graph of each constraint's force at each step of the resolution
    Data<std::map < std::string, sofa::type::vector<SReal> > > d_graphViolations; ///< Graph of each constraint's violation at each step of the resolution

    Data<int> d_currentNumConstraints; ///< OUTPUT: current number of constraints
    Data<int> d_currentNumConstraintGroups; ///< OUTPUT: current number of constraints
    Data<int> d_currentIterations; ///< OUTPUT: current number of constraint groups
    Data<SReal> d_currentError; ///< OUTPUT: current error
    Data<bool> d_reverseAccumulateOrder; ///< True to accumulate constraints from nodes in reversed order (can be necessary when using multi-mappings or interaction constraints not following the node hierarchy)
    Data<type::vector< SReal >> d_constraintForces; ///< OUTPUT: The Data constraintForces is used to provide the intensities of constraint forces in the simulation. The user can easily check the constraint forces from the GenericConstraint component interface.
    Data<bool> d_computeConstraintForces; ///< The indices of the constraintForces to store in the constraintForce data field.

    sofa::core::MultiVecDerivId getLambda() const override;
    sofa::core::MultiVecDerivId getDx() const override;

protected:

    void clearConstraintProblemLocks();

    static constexpr auto CP_BUFFER_SIZE = 10;
    sofa::type::fixed_array<GenericConstraintProblem, CP_BUFFER_SIZE> m_cpBuffer;
    sofa::type::fixed_array<bool, CP_BUFFER_SIZE> m_cpIsLocked;
    GenericConstraintProblem *current_cp, *last_cp;
    SOFA_ATTRIBUTE_DISABLED__GENERICCONSTRAINTSOLVER_CONSTRAINTCORRECTIONS() DeprecatedAndRemoved constraintCorrections; //use ConstraintSolverImpl::l_constraintCorrections instead

    sofa::core::MultiVecDerivId m_lambdaId;
    sofa::core::MultiVecDerivId m_dxId;

    void buildSystem_matrixFree(unsigned int numConstraints);

    // Explicitly compute the compliance matrix projected in the constraint space
    void buildSystem_matrixAssembly(const core::ConstraintParams *cParams);

private:

    struct ComplianceWrapper
    {
        using ComplianceMatrixType = sofa::linearalgebra::LPtrFullMatrix<SReal>;

        ComplianceWrapper(ComplianceMatrixType& complianceMatrix, bool isMultiThreaded)
        : m_isMultiThreaded(isMultiThreaded), m_complianceMatrix(complianceMatrix) {}

        ComplianceMatrixType& matrix();

        void assembleMatrix() const;

    private:
        bool m_isMultiThreaded { false };
        ComplianceMatrixType& m_complianceMatrix;
        std::unique_ptr<ComplianceMatrixType> m_threadMatrix;
    };


    sofa::type::vector<core::behavior::BaseConstraintCorrection*> filteredConstraintCorrections() const;

    void computeAndApplyMotionCorrection(const core::ConstraintParams* cParams, GenericConstraintSolver::MultiVecId res1, GenericConstraintSolver::MultiVecId res2) const;
    void applyMotionCorrection(
        const core::ConstraintParams* cParams,
        MultiVecId res1, MultiVecId res2,
        core::behavior::BaseConstraintCorrection* constraintCorrection) const;
    void storeConstraintLambdas(const core::ConstraintParams* cParams);

};

} //namespace sofa::component::constraint::lagrangian::solver
