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
    : d_maxIt(initData(&d_maxIt, 1000, "maxIterations", "maximal number of iterations of iterative algorithm"))
    , d_tolerance(initData(&d_tolerance, 0.001_sreal, "tolerance", "residual error threshold for termination of the Gauss-Seidel algorithm"))
    , d_sor(initData(&d_sor, 1.0_sreal, "sor", "Successive Over Relaxation parameter (0-2)"))
    , d_regularizationTerm(initData(&d_regularizationTerm, 0.0_sreal, "regularizationTerm", "Add regularization factor times the identity matrix to the compliance W when solving constraints"))
    , d_scaleTolerance(initData(&d_scaleTolerance, true, "scaleTolerance", "Scale the error tolerance with the number of constraints"))
    , d_allVerified(initData(&d_allVerified, false, "allVerified", "All constraints must be verified (each constraint's error < tolerance)"))
    , d_computeGraphs(initData(&d_computeGraphs, false, "computeGraphs", "Compute graphs of errors and forces during resolution"))
    , d_graphErrors(initData(&d_graphErrors, "graphErrors", "Sum of the constraints' errors at each iteration"))
    , d_graphConstraints(initData(&d_graphConstraints, "graphConstraints", "Graph of each constraint's error at the end of the resolution"))
    , d_graphForces(initData(&d_graphForces, "graphForces", "Graph of each constraint's force at each step of the resolution"))
    , d_graphViolations(initData(&d_graphViolations, "graphViolations", "Graph of each constraint's violation at each step of the resolution"))
    , d_currentNumConstraints(initData(&d_currentNumConstraints, 0, "currentNumConstraints", "OUTPUT: current number of constraints"))
    , d_currentNumConstraintGroups(initData(&d_currentNumConstraintGroups, 0, "currentNumConstraintGroups", "OUTPUT: current number of constraints"))
    , d_currentIterations(initData(&d_currentIterations, 0, "currentIterations", "OUTPUT: current number of constraint groups"))
    , d_currentError(initData(&d_currentError, 0.0_sreal, "currentError", "OUTPUT: current error"))
    , d_constraintForces(initData(&d_constraintForces,"constraintForces","OUTPUT: constraint forces (stored only if computeConstraintForces=True)"))
    , d_computeConstraintForces(initData(&d_computeConstraintForces,false,
                                        "computeConstraintForces",
                                        "enable the storage of the constraintForces."))
    , last_cp(nullptr)
{
    addAlias(&d_maxIt, "maxIt");

    d_graphErrors.setWidget("graph");
    d_graphErrors.setGroup("Graph");

    d_graphConstraints.setWidget("graph");
    d_graphConstraints.setGroup("Graph");

    d_graphForces.setWidget("graph_linear");
    d_graphForces.setGroup("Graph2");

    d_graphViolations.setWidget("graph_linear");
    d_graphViolations.setGroup("Graph2");

    d_currentNumConstraints.setReadOnly(true);
    d_currentNumConstraints.setGroup("Stats");
    d_currentNumConstraintGroups.setReadOnly(true);
    d_currentNumConstraintGroups.setGroup("Stats");
    d_currentIterations.setReadOnly(true);
    d_currentIterations.setGroup("Stats");
    d_currentError.setReadOnly(true);
    d_currentError.setGroup("Stats");

    d_maxIt.setRequired(true);
    d_tolerance.setRequired(true);
}

GenericConstraintSolver::~GenericConstraintSolver()
{
    for (unsigned i=0; i< CP_BUFFER_SIZE; ++i)
    {
        delete m_cpBuffer[i];
    }
}



void GenericConstraintSolver::  init()
{
    this->initializeConstraintProblems();
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
}

void GenericConstraintSolver::initializeConstraintProblems()
{
    for (unsigned i=0; i< CP_BUFFER_SIZE; ++i)
    {
        m_cpBuffer[i] = new GenericConstraintProblem(this);
    }
    current_cp = m_cpBuffer[0];
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

    //clear and/or resize based on the number of constraints
    current_cp->clear(numConstraints);

    getConstraintViolation(cParams, &current_cp->dFree);

    {
        // creates constraint-specific objects used for the constraint resolution
        // in a Gauss-Seidel algorithm
        SCOPED_TIMER("Get Constraint Resolutions");
        MechanicalGetConstraintResolutionVisitor(cParams, current_cp->constraintsResolutions).execute(getContext());
    }



    this->doBuildSystem(cParams, current_cp, numConstraints);

    return true;
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
    current_cp->tolerance = d_tolerance.getValue();
    current_cp->maxIterations = d_maxIt.getValue();
    current_cp->scaleTolerance = d_scaleTolerance.getValue();
    current_cp->allVerified = d_allVerified.getValue();
    current_cp->sor = d_sor.getValue();

    if (notMuted())
    {
        std::stringstream tmp;
        tmp << "---> Before Resolution" << msgendl  ;
        printLCP(tmp, current_cp->getDfree(), current_cp->getW(), current_cp->getF(), current_cp->getDimension(), true);

        msg_info() << tmp.str() ;
    }

    this->doSolve(current_cp, 0.0);


    this->d_currentError.setValue(current_cp->currentError);
    this->d_currentIterations.setValue(current_cp->currentIterations);
    this->d_currentNumConstraints.setValue(current_cp->getNumConstraints());
    this->d_currentNumConstraintGroups.setValue(current_cp->getNumConstraintGroups());

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
                SCOPED_TIMER("doComputeCorrection");
                constraintCorrection->computeMotionCorrectionFromLambda(cParams, this->getDx(), &current_cp->f);
            }

            SCOPED_TIMER("doApplyCorrection");
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
        GenericConstraintProblem* p = m_cpBuffer[i];
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

void GenericConstraintSolver::addRegularization(linearalgebra::BaseMatrix& W, const SReal regularization)
{
    if (regularization>std::numeric_limits<SReal>::epsilon())
    {
        for (int i=0; i<W.rowSize(); ++i)
        {
            W.add(i,i,regularization);
        }
    }
}




} //namespace sofa::component::constraint::lagrangian::solver
