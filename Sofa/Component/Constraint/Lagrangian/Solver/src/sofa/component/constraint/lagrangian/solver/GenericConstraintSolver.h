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
    typedef std::vector<core::behavior::BaseConstraintCorrection*> list_cc;
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
    void buildSystem_matrixFree(unsigned int numConstraints);
    void buildSystem_matrixAssembly(const core::ConstraintParams *cParams);
    void rebuildSystem(SReal massFactor, SReal forceFactor) override;
    bool solveSystem(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;
    bool applyCorrection(const core::ConstraintParams * /*cParams*/, MultiVecId res1, MultiVecId res2=MultiVecId::null()) override;
    void computeResidual(const core::ExecParams* /*params*/) override;
    ConstraintProblem* getConstraintProblem() override;
    void lockConstraintProblem(sofa::core::objectmodel::BaseObject* from, ConstraintProblem* p1, ConstraintProblem* p2 = nullptr) override;
    void removeConstraintCorrection(core::behavior::BaseConstraintCorrection *s) override;

    Data< sofa::helper::OptionsGroup > d_resolutionMethod; ///< Method used to solve the constraint problem, among: \"ProjectedGaussSeidel\", \"UnbuiltGaussSeidel\" or \"for NonsmoothNonlinearConjugateGradient\"

    Data<int> maxIt; ///< maximal number of iterations of the Gauss-Seidel algorithm
    Data<SReal> tolerance; ///< residual error threshold for termination of the Gauss-Seidel algorithm
    Data<SReal> sor; ///< Successive Over Relaxation parameter (0-2)
    Data<bool> scaleTolerance; ///< Scale the error tolerance with the number of constraints
    Data<bool> allVerified; ///< All contraints must be verified (each constraint's error < tolerance)
    Data<int> d_newtonIterations; ///< Maximum iteration number of Newton (for the NNCG solver only)
    Data<bool> d_multithreading; ///< Compliances built concurrently
    Data<bool> computeGraphs; ///< Compute graphs of errors and forces during resolution
    Data<std::map < std::string, sofa::type::vector<SReal> > > graphErrors; ///< Sum of the constraints' errors at each iteration
    Data<std::map < std::string, sofa::type::vector<SReal> > > graphConstraints; ///< Graph of each constraint's error at the end of the resolution
    Data<std::map < std::string, sofa::type::vector<SReal> > > graphForces; ///< Graph of each constraint's force at each step of the resolution
    Data<std::map < std::string, sofa::type::vector<SReal> > > graphViolations; ///< Graph of each constraint's violation at each step of the resolution

    Data<int> currentNumConstraints; ///< OUTPUT: current number of constraints
    Data<int> currentNumConstraintGroups; ///< OUTPUT: current number of constraints
    Data<int> currentIterations; ///< OUTPUT: current number of constraint groups
    Data<SReal> currentError; ///< OUTPUT: current error
    Data<bool> reverseAccumulateOrder; ///< True to accumulate constraints from nodes in reversed order (can be necessary when using multi-mappings or interaction constraints not following the node hierarchy)
    Data<type::vector< SReal >> d_constraintForces; ///< OUTPUT: The Data constraintForces is used to provide the intensities of constraint forces in the simulation. The user can easily check the constraint forces from the GenericConstraint component interface.
    Data<bool> d_computeConstraintForces; ///< The indices of the constraintForces to store in the constraintForce data field.

    SOFA_ATTRIBUTE_DISABLED__GENERICCONSTRAINTSOLVER_DATA("Data schemeCorrection was unused therefore removed.")
    DeprecatedAndRemoved schemeCorrection; ///< Apply new scheme where compliance is progressively corrected
    SOFA_ATTRIBUTE_DISABLED__GENERICCONSTRAINTSOLVER_DATA("Make the \"unbuild\" option as an option group \"resolutionMethod\".")
    Data<bool> unbuilt; ///< Compliance is not fully built  (for the PGS solver only)
    //SOFA_ATTRIBUTE_DISABLED__GENERICCONSTRAINTSOLVER_DATA("Make the \"unbuild\" option as an option group \"resolutionMethod\".")
    void parse( sofa::core::objectmodel::BaseObjectDescription* arg ) override
    {
        Inherit1::parse(arg);
        if (arg->getAttribute("unbuilt"))
        {
            msg_warning() << "String data \"unbuilt\" is now an option group \"resolutionMethod\" (PR #3053)" << msgendl << "Use: resolutionMethod=\"UnbuildGaussSeidel\"";
        }
    }

    sofa::core::MultiVecDerivId getLambda() const override;
    sofa::core::MultiVecDerivId getDx() const override;

protected:

    void clearConstraintProblemLocks();

    enum { CP_BUFFER_SIZE = 10 };
    sofa::type::fixed_array<GenericConstraintProblem,CP_BUFFER_SIZE> m_cpBuffer;
    sofa::type::fixed_array<bool,CP_BUFFER_SIZE> m_cpIsLocked;
    GenericConstraintProblem *current_cp, *last_cp;
    type::vector<core::behavior::BaseConstraintCorrection*> constraintCorrections;
    type::vector<bool> constraintCorrectionIsActive; // for each constraint correction, a boolean that is false if the parent node is sleeping


    sofa::core::objectmodel::BaseContext *context { nullptr };

    sofa::core::MultiVecDerivId m_lambdaId;
    sofa::core::MultiVecDerivId m_dxId;

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

};

} //namespace sofa::component::constraint::lagrangian::solver
