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

#include <sofa/component/constraint/lagrangian/solver/GenericConstraintSolver.h>
#include <sofa/core/behavior/ConstraintResolution.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/component/constraint/lagrangian/solver/visitors/ConstraintStoreLambdaVisitor.h>
#include <sofa/core/behavior/MultiVec.h>
#include <sofa/simulation/DefaultTaskScheduler.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/ParallelForEach.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalVOpVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalVOpVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalProjectJacobianMatrixVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalProjectJacobianMatrixVisitor;

namespace sofa::component::constraint::lagrangian::solver
{

namespace
{

using sofa::helper::WriteOnlyAccessor;
using sofa::core::objectmodel::Data;
using sofa::core::ConstraintParams;

template< typename TMultiVecId >
void clearMultiVecId(sofa::core::objectmodel::BaseContext* ctx, const sofa::core::ConstraintParams* cParams, const TMultiVecId& vid)
{
    MechanicalVOpVisitor clearVisitor(cParams, vid, core::ConstMultiVecDerivId::null(), core::ConstMultiVecDerivId::null(), 1.0);
    clearVisitor.setMapped(true);
    ctx->executeVisitor(&clearVisitor);
}

}

GenericConstraintSolver::GenericConstraintSolver()
    : d_resolutionMethod( initData(&d_resolutionMethod, "resolutionMethod", "Method used to solve the constraint problem, among: \"ProjectedGaussSeidel\", \"UnbuiltGaussSeidel\" or \"for NonsmoothNonlinearConjugateGradient\""))
    , maxIt( initData(&maxIt, 1000, "maxIterations", "maximal number of iterations of the Gauss-Seidel algorithm"))
    , tolerance( initData(&tolerance, 0.001_sreal, "tolerance", "residual error threshold for termination of the Gauss-Seidel algorithm"))
    , sor( initData(&sor, 1.0_sreal, "sor", "Successive Over Relaxation parameter (0-2)"))
    , scaleTolerance( initData(&scaleTolerance, true, "scaleTolerance", "Scale the error tolerance with the number of constraints"))
    , allVerified( initData(&allVerified, false, "allVerified", "All contraints must be verified (each constraint's error < tolerance)"))
    , d_newtonIterations(initData(&d_newtonIterations, 100, "newtonIterations", "Maximum iteration number of Newton (for the NonsmoothNonlinearConjugateGradient solver only)"))
    , d_multithreading(initData(&d_multithreading, false, "multithreading", "Build compliances concurrently"))
    , computeGraphs(initData(&computeGraphs, false, "computeGraphs", "Compute graphs of errors and forces during resolution"))
    , graphErrors( initData(&graphErrors,"graphErrors","Sum of the constraints' errors at each iteration"))
    , graphConstraints( initData(&graphConstraints,"graphConstraints","Graph of each constraint's error at the end of the resolution"))
    , graphForces( initData(&graphForces,"graphForces","Graph of each constraint's force at each step of the resolution"))
    , graphViolations( initData(&graphViolations, "graphViolations", "Graph of each constraint's violation at each step of the resolution"))
    , currentNumConstraints(initData(&currentNumConstraints, 0, "currentNumConstraints", "OUTPUT: current number of constraints"))
    , currentNumConstraintGroups(initData(&currentNumConstraintGroups, 0, "currentNumConstraintGroups", "OUTPUT: current number of constraints"))
    , currentIterations(initData(&currentIterations, 0, "currentIterations", "OUTPUT: current number of constraint groups"))
    , currentError(initData(&currentError, 0.0_sreal, "currentError", "OUTPUT: current error"))
    , reverseAccumulateOrder(initData(&reverseAccumulateOrder, false, "reverseAccumulateOrder", "True to accumulate constraints from nodes in reversed order (can be necessary when using multi-mappings or interaction constraints not following the node hierarchy)"))
    , d_constraintForces(initData(&d_constraintForces,"constraintForces","OUTPUT: constraint forces (stored only if computeConstraintForces=True)"))
    , d_computeConstraintForces(initData(&d_computeConstraintForces,false,
                                        "computeConstraintForces",
                                        "enable the storage of the constraintForces."))
    , current_cp(&m_cpBuffer[0])
    , last_cp(nullptr)
{
    sofa::helper::OptionsGroup m_newoptiongroup{"ProjectedGaussSeidel","UnbuiltGaussSeidel", "NonsmoothNonlinearConjugateGradient"};
    m_newoptiongroup.setSelectedItem("ProjectedGaussSeidel");
    d_resolutionMethod.setValue(m_newoptiongroup);

    addAlias(&maxIt, "maxIt");

    graphErrors.setWidget("graph");
    graphErrors.setGroup("Graph");

    graphConstraints.setWidget("graph");
    graphConstraints.setGroup("Graph");

    graphForces.setWidget("graph_linear");
    graphForces.setGroup("Graph2");

    graphViolations.setWidget("graph_linear");
    graphViolations.setGroup("Graph2");

    currentNumConstraints.setReadOnly(true);
    currentNumConstraints.setGroup("Stats");
    currentNumConstraintGroups.setReadOnly(true);
    currentNumConstraintGroups.setGroup("Stats");
    currentIterations.setReadOnly(true);
    currentIterations.setGroup("Stats");
    currentError.setReadOnly(true);
    currentError.setGroup("Stats");

    maxIt.setRequired(true);
    tolerance.setRequired(true);
}

GenericConstraintSolver::~GenericConstraintSolver()
{}

void GenericConstraintSolver::init()
{
    ConstraintSolverImpl::init();

    simulation::common::VectorOperations vop(sofa::core::execparams::defaultInstance(), this->getContext());
    {
        sofa::core::behavior::MultiVecDeriv lambda(&vop, m_lambdaId);
        lambda.realloc(&vop,false,true, core::VecIdProperties{"lambda", GetClass()->className});
        m_lambdaId = lambda.id();
    }
    {
        sofa::core::behavior::MultiVecDeriv dx(&vop, m_dxId);
        dx.realloc(&vop,false,true, core::VecIdProperties{"constraint_dx", GetClass()->className});
        m_dxId = dx.id();
    }

    if(d_multithreading.getValue())
    {
        simulation::MainTaskSchedulerFactory::createInRegistry()->init();
    }

    if(d_newtonIterations.isSet())
    {
        if (d_resolutionMethod.getValue().getSelectedId() != 2)
        {
            msg_warning() << "data \"newtonIterations\" is not only taken into account when using the NonsmoothNonlinearConjugateGradient solver";
        }
    }
}

void GenericConstraintSolver::cleanup()
{
    simulation::common::VectorOperations vop(sofa::core::execparams::defaultInstance(), this->getContext());
    vop.v_free(m_lambdaId, false, true);
    vop.v_free(m_dxId, false, true);
    sofa::component::constraint::lagrangian::solver::ConstraintSolverImpl::cleanup();
}

bool GenericConstraintSolver::prepareStates(const core::ConstraintParams *cParams, MultiVecId /*res1*/, MultiVecId /*res2*/)
{
    last_cp = current_cp;

    clearConstraintProblemLocks(); // NOTE: this assumes we solve only one constraint problem per step

    simulation::common::VectorOperations vop(cParams, this->getContext());

    {
        sofa::core::behavior::MultiVecDeriv lambda(&vop, m_lambdaId);
        lambda.realloc(&vop,false,true, core::VecIdProperties{"lambda", GetClass()->className});
        m_lambdaId = lambda.id();

        clearMultiVecId(getContext(), cParams, m_lambdaId);
    }

    {
        sofa::core::behavior::MultiVecDeriv dx(&vop, m_dxId);
        dx.realloc(&vop,false,true, core::VecIdProperties{"constraint_dx", GetClass()->className});
        m_dxId = dx.id();

        clearMultiVecId(getContext(), cParams, m_dxId);

    }

    return true;
}

bool GenericConstraintSolver::buildSystem(const core::ConstraintParams *cParams, MultiVecId res1, MultiVecId res2)
{
    SOFA_UNUSED(res1);
    SOFA_UNUSED(res2);

    const unsigned int numConstraints = buildConstraintMatrix(cParams);
    sofa::helper::AdvancedTimer::valSet("numConstraints", numConstraints);

    // suppress the constraints that are on DOFS currently concerned by projective constraint
    applyProjectiveConstraintOnConstraintMatrix(cParams);

    current_cp->clear(numConstraints);

    getConstraintViolation(cParams, &current_cp->dFree);

    {
        // creates constraint-specific objects used for the constraint resolution
        // in a Gauss-Seidel algorithm
        SCOPED_TIMER("Get Constraint Resolutions");
        MechanicalGetConstraintResolutionVisitor(cParams, current_cp->constraintsResolutions).execute(getContext());
    }

    // Resolution depending on the method selected
    switch ( d_resolutionMethod.getValue().getSelectedId() )
    {
        case 0: // ProjectedGaussSeidel
        case 2: // NonsmoothNonlinearConjugateGradient
        {
            buildSystem_matrixAssembly(cParams);
            break;
        }
        case 1: // UnbuiltGaussSeidel
        {
            buildSystem_matrixFree(numConstraints);
            break;
        }
        default:
            msg_error() << "Wrong \"resolutionMethod\" given";
    }

    return true;
}

void GenericConstraintSolver::buildSystem_matrixFree(unsigned int numConstraints)
{
    for (const auto& cc : l_constraintCorrections)
    {
        if (!cc->isActive()) continue;

        current_cp->constraints_sequence.resize(numConstraints);
        std::iota(current_cp->constraints_sequence.begin(), current_cp->constraints_sequence.end(), 0);

        // some constraint corrections (e.g LinearSolverConstraintCorrection)
        // can change the order of the constraints, to optimize later computations
        cc->resetForUnbuiltResolution(current_cp->getF(), current_cp->constraints_sequence);
    }

    sofa::linearalgebra::SparseMatrix<SReal>* Wdiag = &current_cp->Wdiag;
    Wdiag->resize(numConstraints, numConstraints);

    // for each contact, the constraint corrections that are involved with the contact are memorized
    current_cp->cclist_elems.clear();
    current_cp->cclist_elems.resize(numConstraints);
    const int nbCC = l_constraintCorrections.size();
    for (unsigned int i = 0; i < numConstraints; i++)
        current_cp->cclist_elems[i].resize(nbCC, nullptr);

    unsigned int nbObjects = 0;
    for (unsigned int c_id = 0; c_id < numConstraints;)
    {
        bool foundCC = false;
        nbObjects++;
        const unsigned int l = current_cp->constraintsResolutions[c_id]->getNbLines();

        for (unsigned int j = 0; j < l_constraintCorrections.size(); j++)
        {
            core::behavior::BaseConstraintCorrection* cc = l_constraintCorrections[j];
            if (!cc->isActive()) continue;
            if (cc->hasConstraintNumber(c_id))
            {
                current_cp->cclist_elems[c_id][j] = cc;
                cc->getBlockDiagonalCompliance(Wdiag, c_id, c_id + l - 1);
                foundCC = true;
            }
        }

        msg_error_when(!foundCC) << "No constraintCorrection found for constraint" << c_id ;

        SReal** w =  current_cp->getW();
        for(unsigned int m = c_id; m < c_id + l; m++)
            for(unsigned int n = c_id; n < c_id + l; n++)
                w[m][n] = Wdiag->element(m, n);

        c_id += l;
    }

    current_cp->change_sequence = false;
    if(current_cp->constraints_sequence.size() == nbObjects)
        current_cp->change_sequence=true;
}

GenericConstraintSolver::ComplianceWrapper::ComplianceMatrixType& GenericConstraintSolver::
ComplianceWrapper::matrix()
{
    if (m_isMultiThreaded)
    {
        if (!m_threadMatrix)
        {
            m_threadMatrix = std::make_unique<ComplianceMatrixType>();
            m_threadMatrix->resize(m_complianceMatrix.rowSize(), m_complianceMatrix.colSize());
        }
        return *m_threadMatrix;
    }
    return m_complianceMatrix;
}

void GenericConstraintSolver::ComplianceWrapper::assembleMatrix() const
{
    if (m_threadMatrix)
    {
        for (linearalgebra::BaseMatrix::Index j = 0; j < m_threadMatrix->rowSize(); ++j)
        {
            for (linearalgebra::BaseMatrix::Index l = 0; l < m_threadMatrix->colSize(); ++l)
            {
                m_complianceMatrix.add(j, l, m_threadMatrix->element(j,l));
            }
        }
    }
}

void GenericConstraintSolver::buildSystem_matrixAssembly(const core::ConstraintParams *cParams)
{
    SCOPED_TIMER_VARNAME(getComplianceTimer, "Get Compliance");
    dmsg_info() <<" computeCompliance in "  << l_constraintCorrections.size()<< " constraintCorrections" ;

    const bool multithreading = d_multithreading.getValue();

    const simulation::ForEachExecutionPolicy execution = multithreading ?
        simulation::ForEachExecutionPolicy::PARALLEL :
        simulation::ForEachExecutionPolicy::SEQUENTIAL;

    simulation::TaskScheduler* taskScheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler);

    //Used to prevent simultaneous accesses to the main compliance matrix
    std::mutex mutex;

    //Visits all constraint corrections to compute the compliance matrix projected
    //in the constraint space.
    simulation::forEachRange(execution, *taskScheduler, l_constraintCorrections.begin(), l_constraintCorrections.end(),
        [&cParams, this, &multithreading, &mutex](const auto& range)
        {
            ComplianceWrapper compliance(current_cp->W, multithreading);

            for (auto it = range.start; it != range.end; ++it)
            {
                core::behavior::BaseConstraintCorrection* cc = *it;
                if (cc->isActive())
                {
                    cc->addComplianceInConstraintSpace(cParams, &compliance.matrix());
                }
            }

            std::lock_guard guard(mutex);
            compliance.assembleMatrix();
        });

    dmsg_info() << " computeCompliance_done "  ;
}

void GenericConstraintSolver::rebuildSystem(const SReal massFactor, const SReal forceFactor)
{
    for (const auto& cc : l_constraintCorrections)
    {
        if (cc->isActive())
        {
            cc->rebuildSystem(massFactor, forceFactor);
        }
    }
}

void printLCP(std::ostream& file, SReal *q, SReal **M, SReal *f, int dim, bool printMatrix = true)
{
    file.precision(9);

    // print LCP matrix
    if (printMatrix)
    {
        file << msgendl << " W = [";
        for (int row = 0; row < dim; row++)
        {
            for (int col = 0; col < dim; col++)
            {
                file << "\t" << M[row][col];
            }
            file << msgendl;
        }
        file << "      ];" << msgendl << msgendl;
    }

    // print q
    file << " delta = [";
    for (int i = 0; i < dim; i++)
    {
        file << "\t" << q[i];
    }
    file << "      ];" << msgendl << msgendl;

    // print f
    file << " lambda = [";
    for (int i = 0; i < dim; i++)
    {
        file << "\t" << f[i];
    }
    file << "      ];" << msgendl << msgendl;
}

bool GenericConstraintSolver::solveSystem(const core::ConstraintParams * /*cParams*/, MultiVecId /*res1*/, MultiVecId /*res2*/)
{
    current_cp->tolerance = tolerance.getValue();
    current_cp->maxIterations = maxIt.getValue();
    current_cp->scaleTolerance = scaleTolerance.getValue();
    current_cp->allVerified = allVerified.getValue();
    current_cp->sor = sor.getValue();


    // Resolution depending on the method selected
    switch ( d_resolutionMethod.getValue().getSelectedId() )
    {
        // ProjectedGaussSeidel
        case 0: {
            if (notMuted())
            {
                std::stringstream tmp;
                tmp << "---> Before Resolution" << msgendl  ;
                printLCP(tmp, current_cp->getDfree(), current_cp->getW(), current_cp->getF(), current_cp->getDimension(), true);

                msg_info() << tmp.str() ;
            }
            SCOPED_TIMER_VARNAME(gaussSeidelTimer, "ConstraintsGaussSeidel");
            current_cp->gaussSeidel(0, this);
            break;
        }
        // UnbuiltGaussSeidel
        case 1: {
            SCOPED_TIMER_VARNAME(unbuiltGaussSeidelTimer, "ConstraintsUnbuiltGaussSeidel");
            current_cp->unbuiltGaussSeidel(0, this);
            break;
        }
        // NonsmoothNonlinearConjugateGradient
        case 2: {
            current_cp->NNCG(this, d_newtonIterations.getValue());
            break;
        }
        default:
            msg_error() << "Wrong \"resolutionMethod\" given";
    }


    this->currentError.setValue(current_cp->currentError);
    this->currentIterations.setValue(current_cp->currentIterations);
    this->currentNumConstraints.setValue(current_cp->getNumConstraints());
    this->currentNumConstraintGroups.setValue(current_cp->getNumConstraintGroups());

    if(notMuted())
    {
        std::stringstream tmp;
        tmp << "---> After Resolution" << msgendl;
        printLCP(tmp, current_cp->_d.ptr(), current_cp->getW(), current_cp->getF(), current_cp->getDimension(), false);
        msg_info() << tmp.str() ;
    }

    if(d_computeConstraintForces.getValue())
    {
        WriteOnlyAccessor<Data<type::vector<SReal>>> constraints = d_constraintForces;
        constraints.resize(current_cp->getDimension());
        for(int i=0; i<current_cp->getDimension(); i++)
        {
            constraints[i] = current_cp->getF()[i];
        }
    }

    return true;
}

void GenericConstraintSolver::computeResidual(const core::ExecParams* eparam)
{
    for (const auto& cc : l_constraintCorrections)
    {
        cc->computeResidual(eparam,&current_cp->f);
    }
}

sofa::type::vector<core::behavior::BaseConstraintCorrection*> GenericConstraintSolver::filteredConstraintCorrections() const
{
    sofa::type::vector<core::behavior::BaseConstraintCorrection*> filteredConstraintCorrections;
    filteredConstraintCorrections.reserve(l_constraintCorrections.size());
    std::copy_if(l_constraintCorrections.begin(), l_constraintCorrections.end(),
         std::back_inserter(filteredConstraintCorrections),
         [](const auto& constraintCorrection)
         {
             return constraintCorrection &&
                     !constraintCorrection->getContext()->isSleeping() &&
                     constraintCorrection->isActive();
         });
    return filteredConstraintCorrections;
}

void GenericConstraintSolver::applyMotionCorrection(
    const core::ConstraintParams* cParams,
    MultiVecId res1, MultiVecId res2,
    core::behavior::BaseConstraintCorrection* constraintCorrection) const
{
    if (cParams->constOrder() == sofa::core::ConstraintOrder::POS_AND_VEL)
    {
        constraintCorrection->applyMotionCorrection(cParams, core::MultiVecCoordId{res1}, core::MultiVecDerivId{res2}, cParams->dx(), this->getDx() );
    }
    else if (cParams->constOrder() == sofa::core::ConstraintOrder::POS)
    {
        constraintCorrection->applyPositionCorrection(cParams, core::MultiVecCoordId{res1}, cParams->dx(), this->getDx());
    }
    else if (cParams->constOrder() == sofa::core::ConstraintOrder::VEL)
    {
        constraintCorrection->applyVelocityCorrection(cParams, core::MultiVecDerivId{res1}, cParams->dx(), this->getDx() );
    }
}

void GenericConstraintSolver::computeAndApplyMotionCorrection(const core::ConstraintParams* cParams, MultiVecId res1, MultiVecId res2) const
{
    SCOPED_TIMER("Compute And Apply Motion Correction");

    static constexpr auto supportedCorrections = {
        sofa::core::ConstraintOrder::POS_AND_VEL,
        sofa::core::ConstraintOrder::POS,
        sofa::core::ConstraintOrder::VEL
    };

    if (std::find(supportedCorrections.begin(), supportedCorrections.end(), cParams->constOrder()) != supportedCorrections.end())
    {
        for (const auto& constraintCorrection : filteredConstraintCorrections())
        {
            {
                SCOPED_TIMER("ComputeCorrection");
                constraintCorrection->computeMotionCorrectionFromLambda(cParams, this->getDx(), &current_cp->f);
            }

            SCOPED_TIMER("ApplyCorrection");
            applyMotionCorrection(cParams, res1, res2, constraintCorrection);
        }
    }
}

void GenericConstraintSolver::storeConstraintLambdas(const core::ConstraintParams* cParams)
{
    SCOPED_TIMER("Store Constraint Lambdas");

    /// Some constraint correction schemes may have written the constraint motion space lambda in the lambdaId VecId.
    /// In order to be sure that we are not accumulating things twice, we need to clear.
    clearMultiVecId(getContext(), cParams, m_lambdaId);

    /// Store lambda and accumulate.
    ConstraintStoreLambdaVisitor v(cParams, &current_cp->f);
    this->getContext()->executeVisitor(&v);
}


bool GenericConstraintSolver::applyCorrection(const core::ConstraintParams *cParams, MultiVecId res1, MultiVecId res2)
{
    computeAndApplyMotionCorrection(cParams, res1, res2);
    storeConstraintLambdas(cParams);

    return true;
}


ConstraintProblem* GenericConstraintSolver::getConstraintProblem()
{
    return last_cp;
}

void GenericConstraintSolver::clearConstraintProblemLocks()
{
    std::fill(m_cpIsLocked.begin(), m_cpIsLocked.end(), false);
}

void GenericConstraintSolver::lockConstraintProblem(sofa::core::objectmodel::BaseObject* from, ConstraintProblem* p1, ConstraintProblem* p2)
{
    if( (current_cp != p1) && (current_cp != p2) ) // The current ConstraintProblem is not locked
        return;

    for (unsigned int i = 0; i < CP_BUFFER_SIZE; ++i)
    {
        GenericConstraintProblem* p = &m_cpBuffer[i];
        if (p == p1 || p == p2)
        {
            m_cpIsLocked[i] = true;
        }
        if (!m_cpIsLocked[i]) // ConstraintProblem i is not locked
        {
            current_cp = p;
            return;
        }
    }
    // All constraint problems are locked
    msg_error() << "All constraint problems are locked, request from " << (from ? from->getName() : "nullptr") << " ignored";
}

sofa::core::MultiVecDerivId GenericConstraintSolver::getLambda()  const
{
    return m_lambdaId;
}

sofa::core::MultiVecDerivId GenericConstraintSolver::getDx() const
{
    return m_dxId;
}


int GenericConstraintSolverClass = core::RegisterObject("A Generic Constraint Solver using the Linear Complementarity Problem formulation to solve Constraint based components")
        .add< GenericConstraintSolver >();

} //namespace sofa::component::constraint::lagrangian::solver
