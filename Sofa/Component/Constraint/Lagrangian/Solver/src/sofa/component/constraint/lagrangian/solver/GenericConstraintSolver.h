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

#include <sofa/component/constraint/lagrangian/solver/ConstraintSolverImpl.h>
#include <sofa/core/behavior/BaseConstraintCorrection.h>
#include <sofa/core/behavior/BaseConstraint.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/helper/map.h>

#include <sofa/simulation/CpuTask.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa::component::constraint::lagrangian::solver
{

class GenericConstraintSolver;

class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API GenericConstraintProblem : public ConstraintProblem
{
public:
    sofa::linearalgebra::FullVector<SReal> _d;
    std::vector<core::behavior::ConstraintResolution*> constraintsResolutions;
    bool scaleTolerance, allVerified;
    SReal sor;
    SReal sceneTime;
    SReal currentError;
    int currentIterations;

    // For unbuilt version :
    sofa::linearalgebra::SparseMatrix<SReal> Wdiag;
    std::list<unsigned int> constraints_sequence;
    bool change_sequence;

    typedef std::vector< core::behavior::BaseConstraintCorrection* > ConstraintCorrections;
    typedef std::vector< core::behavior::BaseConstraintCorrection* >::iterator ConstraintCorrectionIterator;

    std::vector< ConstraintCorrections > cclist_elems;


    GenericConstraintProblem() : scaleTolerance(true), allVerified(false), sor(1.0)
      , sceneTime(0.0), currentError(0.0), currentIterations(0)
      , change_sequence(false) {}
    ~GenericConstraintProblem() override { freeConstraintResolutions(); }

    void clear(int nbConstraints) override;
    void freeConstraintResolutions();
    void solveTimed(SReal tol, int maxIt, SReal timeout) override;

    /// Projective Gauss Seidel method building the compliance matrix
    void gaussSeidel(SReal timeout=0, GenericConstraintSolver* solver = nullptr);
    /// Projective Gauss Seidel unbuilt method
    void unbuiltGaussSeidel(SReal timeout=0, GenericConstraintSolver* solver = nullptr);
    /// Method from:
    /// A nonsmooth nonlinear conjugate gradient method for interactive contact force problems
    /// - 2010, Silcowitz, Morten and Niebe, Sarah and Erleben, Kenny
    void NNCG(GenericConstraintSolver* solver = nullptr, int iterationNewton = 1);

    void gaussSeidel_increment(bool measureError, SReal *dfree, SReal *force, SReal **w, SReal tol, SReal *d, int dim, bool& constraintsAreVerified, SReal& error, sofa::type::vector<SReal>& tabErrors) const;
    void result_output(GenericConstraintSolver* solver, SReal *force, SReal error, int iterCount, bool convergence);

    int getNumConstraints();
    int getNumConstraintGroups();

protected:
    sofa::linearalgebra::FullVector<SReal> m_lam;
    sofa::linearalgebra::FullVector<SReal> m_deltaF;
    sofa::linearalgebra::FullVector<SReal> m_deltaF_new;
    sofa::linearalgebra::FullVector<SReal> m_p;

};

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
    void doBaseObjectInit() override;

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
    std::vector<core::behavior::BaseConstraintCorrection*> constraintCorrections;
    std::vector<char> constraintCorrectionIsActive; // for each constraint correction, a boolean that is false if the parent node is sleeping


    sofa::core::objectmodel::BaseContext *context;

    sofa::core::MultiVecDerivId m_lambdaId;
    sofa::core::MultiVecDerivId m_dxId;

private:

    class ComputeComplianceTask : public simulation::CpuTask
    {
    public:
        ComputeComplianceTask(simulation::CpuTask::Status* status): CpuTask(status) {}
        ~ComputeComplianceTask() override {}

        MemoryAlloc run() final {
            cc->addComplianceInConstraintSpace(&cparams, &W);
            return MemoryAlloc::Stack;
        }

        void set(core::behavior::BaseConstraintCorrection* _cc, core::ConstraintParams _cparams, int dim){
            cc = _cc;
            cparams = _cparams;
            W.resize(dim,dim);
        }

    private:
        core::behavior::BaseConstraintCorrection* cc { nullptr };
        sofa::linearalgebra::LPtrFullMatrix<SReal> W;
        core::ConstraintParams cparams;
        friend class GenericConstraintSolver;
    };
};


class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_SOLVER_API MechanicalGetConstraintResolutionVisitor : public simulation::BaseMechanicalVisitor
{
public:
    MechanicalGetConstraintResolutionVisitor(const core::ConstraintParams* params, std::vector<core::behavior::ConstraintResolution*>& res);

    Result fwdConstraintSet(simulation::Node* node, core::behavior::BaseConstraintSet* cSet) override;

    /// Return a class name for this visitor
    /// Only used for debugging / profiling purposes
    const char* getClassName() const override;

    bool isThreadSafe() const override;
    // This visitor must go through all mechanical mappings, even if isMechanical flag is disabled
    bool stopAtMechanicalMapping(simulation::Node* node, core::BaseMapping* map) override;

#ifdef SOFA_DUMP_VISITOR_INFO
    void setReadWriteVectors() override { }
#endif
private:
    /// Constraint parameters
    const sofa::core::ConstraintParams *cparams;

    std::vector<core::behavior::ConstraintResolution*>& _res;
    unsigned int _offset;
};

} //namespace sofa::component::constraint::lagrangian::solver
