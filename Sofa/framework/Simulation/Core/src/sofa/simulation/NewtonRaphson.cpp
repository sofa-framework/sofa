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
#include <sofa/core/ConstraintParams.h>
#include <sofa/simulation/NewtonRaphson.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/IntegrateVisitor.h>
#include <sofa/simulation/events/AnimateBeginEvent.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/MultiVec.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalVInitVisitor.h>
#include <sofa/simulation/events/AnimateBeginEvent.h>
#include <sofa/simulation/events/AnimateEndEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/UpdateInternalDataVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/events/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalBeginIntegrationVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalEndIntegrationVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalComputeGeometricStiffness.h>
#include <sofa/helper/ScopedAdvancedTimer.h>



namespace sofa::simulation
{
NewtonRaphson::NewtonRaphson()
: d_maxNbIterationsNewton(initData(&d_maxNbIterationsNewton, static_cast<unsigned int>(10) , "maxNbIterationsNewton", "Maximum number of iteration for the Newton algorithm"))
, d_maxNbIterationsLineSearch(initData(&d_maxNbIterationsLineSearch, static_cast<unsigned int>(1), "maxNbIterationsLineSearch", "Maximum number of iteration for the backtracking linesearch algorithm"))
, d_newtonStepSize(initData(&d_newtonStepSize, 1.0_sreal , "newtonStepSize", "Size of the first newton step before the linesearch"))
, d_lineSearchReductionRate(initData(&d_lineSearchReductionRate, 0.5_sreal , "lineSearchReductionRate", "Taken in [0,1[ representing the fraction of diminution of the step done in the backtracking line search (if set to 0.3, the first line search will reduce the step from 1.0 to 0.7)"))
, d_lineSearchArmijoFactor(initData(&d_lineSearchArmijoFactor, 1e-3_sreal , "lineSearchArmijoFactor", "Taken in [0,1[ it represents a tolerance on the residue in term of the linear approximation. e.g., for a value of 0.01, it means we want the solution to decrease the residue as much as 0.01 times the linear approximation in the same direction."))
, d_residueThreshold(initData(&d_residueThreshold, 1e-9_sreal , "residueThreshold", "Threshold under which, the residue is considered to be sufficiently low. Newton algorithm will stop after reaching a lower value"))
, d_currentResidue(initData(&d_currentResidue , "currentResidue", "Current value of the residue"))
, d_alwaysAdvanceNewton(initData(&d_alwaysAdvanceNewton , false, "alwaysAdvanceNewton", "Even if the linesearch didn't find a better solution than the current one, take the best one along the path that is not the current guess."))
, d_performCollisionDetectionEachNewtonStep(initData(&d_performCollisionDetectionEachNewtonStep , false, "performCollisionDetectionEachNewtonStep", "If true, each newton iteration will recompute the collision detection with the updated position from the previous newton iteration."))
, d_solveVelocityConstraintFirst(initData(&d_solveVelocityConstraintFirst , false, "solveVelocityConstraintFirst", "solve separately velocity constraint violations before position constraint violations"))
, d_parallelODESolving(initData(&d_parallelODESolving, false, "parallelODESolving", "If true, solves all the ODEs in parallel during the free motion step."))
, m_solveConstraints(false)
{
}

NewtonRaphson::~NewtonRaphson() = default;

void NewtonRaphson::init()
{
    Inherit1::init();
    if (!l_node)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
    else
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }

    if (!l_constraintSolver)
    {
        l_constraintSolver.set(this->getContext()->get<sofa::core::behavior::ConstraintSolver>(core::objectmodel::BaseContext::SearchDown));
        if (!l_constraintSolver)
        {
            msg_info() << "Constraint solver not found, only weak and projective constraint will be taken into account.";
        }
        else
        {
            msg_info() << "Constraint solver found: '" << l_constraintSolver->getPathName() << "'";
        }
    }

    m_solveConstraints = l_constraintSolver.get() != nullptr;


    if (d_parallelODESolving.getValue())
    {
        initTaskScheduler();
    }
}

void NewtonRaphson::integrate(const core::ExecParams* params, SReal dt)
{
    auto node = dynamic_cast<sofa::simulation::Node*>(this->l_node.get());

    core::ConstraintParams cparams(*params);
    common::VectorOperations vop(params, node);
    common::MechanicalOperations mop(params, getContext());

    SReal startTime = node->getTime();

    auto xResult = sofa::core::vec_id::write_access::position;
    auto vResult = sofa::core::vec_id::write_access::velocity;
    if (m_solveConstraints)
    {
        xResult = sofa::core::vec_id::write_access::freePosition;
        vResult = sofa::core::vec_id::write_access::freeVelocity;
    }

    if (dt == 0)
        dt = node->getDt();

    preSolve(params, &cparams, &vop, &mop, dt, node);
    newtonSolve(params, &cparams, &vop, &mop, dt, xResult, vResult, node);
    postSolve(params, &cparams, &vop, &mop, startTime, dt, node);

}


void NewtonRaphson::preSolve(const core::ExecParams* params, core::ConstraintParams* cparams, common::VectorOperations* vop, common::MechanicalOperations* mop, SReal dt, Node* node)
{

    core::behavior::MultiVecDeriv dx(vop, core::vec_id::write_access::dx);
    dx.realloc(vop, true, true);

    core::behavior::MultiVecDeriv df(vop, core::vec_id::write_access::dforce);
    df.realloc(vop, true, true);

    if (m_solveConstraints)
    {

        core::behavior::MultiVecCoord freePos(vop, core::vec_id::write_access::freePosition );
        core::behavior::MultiVecDeriv freeVel(vop, core::vec_id::write_access::freeVelocity );

        cparams->setX(freePos);
        cparams->setV(freeVel);
        cparams->setDx(l_constraintSolver->getDx());
        cparams->setLambda(l_constraintSolver->getLambda());
        cparams->setOrder(d_solveVelocityConstraintFirst.getValue() ? core::ConstraintOrder::VEL : core::ConstraintOrder::POS_AND_VEL);


        // This solver will work in freePosition and freeVelocity vectors.
        // We need to initialize them if it's not already done.
        {
            SCOPED_TIMER("MechanicalVInitVisitor");
            mechanicalvisitor::MechanicalVInitVisitor< core::V_COORD >(params, core::vec_id::write_access::freePosition, core::vec_id::read_access::position, true).execute(node);
            mechanicalvisitor::MechanicalVInitVisitor< core::V_DERIV >(params, core::vec_id::write_access::freeVelocity, core::vec_id::read_access::velocity, true).execute(node);
        }

        // This animation loop works with lagrangian constraints. Forces derive from the constraints.
        // Therefore we notice the States that they have to consider them in the total accumulation of
        // forces.
        for (auto* state : node->getTreeObjects<sofa::core::BaseState>())
        {
            state->addToTotalForces(cparams->lambda().getId(state));
        }

    }


    {
        SCOPED_TIMER("AnimateBeginEvent");
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }

    // Update the BehaviorModels
    // Required to allow the RayPickInteractor interaction
    dmsg_info() << "updatePos called" ;

    {
        SCOPED_TIMER("UpdatePosition");
        BehaviorUpdatePositionVisitor beh(params, dt);
        node->execute(&beh);
    }

    dmsg_info() << "updatePos performed - updateInternal called" ;

    UpdateInternalDataVisitor iud(params);

    dmsg_info() << "updateInternal called" ;

    {
        SCOPED_TIMER("updateInternalData");
        node->execute(&iud);
    }

    dmsg_info() << "updateInternal performed - beginVisitor called" ;



}


void NewtonRaphson::newtonSolve(const core::ExecParams* params, core::ConstraintParams* cparams, common::VectorOperations* vop, common::MechanicalOperations* mop,  SReal dt, sofa::core::MultiVecCoordId xId, sofa::core::MultiVecDerivId vId, Node* node)
{
    // MechanicalBeginIntegrationVisitor
    mechanicalvisitor::MechanicalBeginIntegrationVisitor beginVisitor(params, dt);
    node->execute(&beginVisitor);

    dmsg_info() << "beginVisitor performed - SolveVisitor for freeMotion is called" ;

    //Constify the data values
    const unsigned maxNewtonIt = d_maxNbIterationsNewton.getValue();
    const unsigned maxLineSearchIt = d_maxNbIterationsLineSearch.getValue();
    const SReal newtonStepSize = d_newtonStepSize.getValue();
    const SReal residueThreshold = d_residueThreshold.getValue();
    const SReal lineSearchReductionRate = d_lineSearchReductionRate.getValue();
    const SReal lineSearchArmijoFactor = d_lineSearchArmijoFactor.getValue();
    const bool performCollisionDetectionEachNewtonStep = d_performCollisionDetectionEachNewtonStep.getValue();

    const bool printLog = f_printLog.getValue();

    IntegrationHelper intHelper(params, dt,  xId, vId, d_parallelODESolving.getValue(), true);
    intHelper.setNode(node);

    //Setup tue integration step
    intHelper.setupIntegrationStep();

    //Compute current residual, useful for static solver to return fast
    //Collision is computed here so that "weak" constraints are accumulated in the RHS
    computeCollision(params);
    //Same for geometrical stiffness
    computeGeometricalStiffness(&mop->mparams, cparams, node);
    intHelper.computeRHS(true);

    SReal oldResidue = intHelper.evaluateResidual();
    SReal newResidue = oldResidue;


    unsigned it = 0;
    while ( it<maxNewtonIt && newResidue>residueThreshold )
    {
        SCOPED_TIMER_VARNAME(step_timer, "NewtonStep");

        const bool firstIt = it==0;

        // If in first iteration, this has already been computed earlier
        if ( ! firstIt )
        {
            if (performCollisionDetectionEachNewtonStep)
                computeCollision(params);

            computeGeometricalStiffness(&mop->mparams, cparams, node);
            intHelper.computeRHS(firstIt);
            oldResidue = intHelper.evaluateResidual();
        }


        double bestresidual = oldResidue;
        double bestalpha = 0.0;

        intHelper.computeLHS(firstIt);
        //Find decrease direction
        intHelper.solveLinearEquation();

        //Setup variables for linesearch
        SReal alpha = newtonStepSize;
        SReal delta = 0.0;

        //Already make a full step
        intHelper.updateStatesFromLinearSolution(alpha, firstIt);
        //TODO is this propagation really required ?
        mop->propagateXAndV(xId, vId); //Need to propagate explicitly to enable recomputation of mapped Forcefileld

        solveConstraints(&mop->mparams, cparams, vop, mop, xId, vId,dt);
        mop->propagateXAndV(xId, vId); //Need to propagate explicitly to enable recomputation of mapped Forcefileld


        intHelper.computeRHS(false);
        newResidue = intHelper.evaluateResidual();

        // Compute the Armijo term
        // !! Careful this approximation holds only when A is symmetric
        // This comes from the fact that if the current residual is f=||r||**2 and the search
        // direction results from Ap = -r, then we have \nabla f = 2*A*r, so the armijo factor can be
        // simplified with \nabla f * p = 2*r^T*A^T*A^{-1}*(-r) = -2*||r||^2  if A is symmetric
        // Otherwise we need to compute 2 * r^{t} * A^{T} * p
        const SReal armijoTerm = - 2 * lineSearchArmijoFactor * oldResidue;

        if (newResidue<bestresidual || d_alwaysAdvanceNewton.getValue())
        {
            bestresidual = newResidue;
            bestalpha = alpha;
        }

        unsigned lineSearchIt = 0;
        while ((newResidue>(oldResidue + alpha*armijoTerm)) && lineSearchIt<maxLineSearchIt )
        {
            //We are backtracking on the same line. Instead of starting from initial position and
            //adding alpha each time, we go back toward the initial position
            delta = alpha*lineSearchReductionRate;
            alpha -= delta;

            intHelper.updateStatesFromLinearSolution(-delta, false);
            mop->propagateXAndV(xId, vId); //Need to propagate explicitly to enable recomputation of mapped Forcefileld

            intHelper.computeRHS(false);
            newResidue = intHelper.evaluateResidual();

            if (newResidue<bestresidual)
            {
                bestresidual = newResidue;
                bestalpha = alpha;
            }

            ++lineSearchIt;
        }

        if (fabs(alpha - bestalpha )> std::numeric_limits<SReal>::epsilon() )
        {
            intHelper.updateStatesFromLinearSolution( bestalpha - alpha, false);
            mop->propagateXAndV(xId, vId); //Need to propagate explicitly to enable recomputation of mapped Forcefileld

            intHelper.computeRHS(false);
            newResidue = intHelper.evaluateResidual();
        }


        if (printLog)
        {
            if (newResidue>oldResidue)
            {
                msg_warning()<<"Newton step increased the residual";
            }
            msg_info()<<"Newton step = "<<it;
            msg_info()<<"Current residue = "<<newResidue<< "   | previous residue = "<<oldResidue ;
            msg_info()<<"Number of line search iterations = "<<lineSearchIt;
        }
        ++it ;
    }
    if (printLog)
    {
        if (newResidue<residueThreshold)
        {
            msg_info()<<"Newton converged to residue "<<newResidue<<" in "<<it<<" steps.";
        }
        else
        {
            msg_warning()<<"Newton didn't converge ! Current residue is "<<newResidue;
        }
    }

    helper::AdvancedTimer::valSet("nb_iterations", it);
    helper::AdvancedTimer::valSet("residual", std::sqrt(newResidue));

    d_currentResidue.setValue(newResidue);

    mechanicalvisitor::MechanicalEndIntegrationVisitor endVisitor(params, dt);
    node->execute(&endVisitor);

}

void NewtonRaphson::postSolve(const core::ExecParams* params, core::ConstraintParams* cparams, common::VectorOperations* vop, common::MechanicalOperations* mop, SReal startTime, SReal dt, Node* node)
{

    auto pos = sofa::core::vec_id::write_access::position;
    auto vel = sofa::core::vec_id::write_access::velocity;
    {
        SCOPED_TIMER("ProjectAndPropagateXAndV");

        mop->projectPositionAndVelocity(pos, vel);
        mop->propagateXAndV(pos, vel);
    }

    node->setTime ( startTime + dt );
    node->execute<UpdateSimulationContextVisitor>(params);  // propagate time

    {
        SCOPED_TIMER("AnimateEndEvent");
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }

    {
        SCOPED_TIMER("UpdateMapping");
        //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
        node->execute<UpdateMappingVisitor>(params);
        {
            UpdateMappingEndEvent ev ( dt );
            PropagateEventVisitor act ( params , &ev );
            node->execute ( act );
        }
    }

    if (d_computeBoundingBox.getValue())
    {
        SCOPED_TIMER("UpdateBBox");
        node->execute<UpdateBoundingBoxVisitor>(params);
    }
}

void NewtonRaphson::computeCollision(const core::ExecParams* params)
{
    helper::ScopedAdvancedTimer collisionDetectionTimer("CollisionDetection");
    Inherit1::computeCollision(params);
}

void NewtonRaphson::computeGeometricalStiffness(const core::MechanicalParams* mparams, const core::ConstraintParams* cparams, Node* node)
{
    SCOPED_TIMER("MechanicalComputeGeometricStiffness");
    mechanicalvisitor::MechanicalComputeGeometricStiffness geometricStiffnessVisitor(mparams, cparams->lambda());
    node->executeVisitor(&geometricStiffnessVisitor);
}

void NewtonRaphson::solveConstraints(const core::MechanicalParams* mparams, const core::ConstraintParams* cparams, common::VectorOperations* vop, common::MechanicalOperations* mop, sofa::core::MultiVecCoordId xId, sofa::core::MultiVecDerivId vId, SReal dt)
{
    SCOPED_TIMER("ConstraintSolver");

    if (cparams->constOrder() == core::ConstraintOrder::VEL )
    {
        l_constraintSolver->solveConstraint(cparams, vId);
        //TODO Why is this even necessary, shouldn't we call the integraitons cheme to perform this integration ?
        vop->v_peq(xId, vId, dt); //position += velocity * dt
    }
    else
    {
        l_constraintSolver->solveConstraint(cparams, xId, vId);
    }

    {
        SCOPED_TIMER("ProjectAndPropagateDx");

        core::behavior::MultiVecDeriv cdx(vop, l_constraintSolver->getDx());
        mop->projectResponse(cdx);
        mop->propagateDx(cdx, true);
    }
}

void registerNewtonRaphson(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("TODO")
        .add<NewtonRaphson>()
        .addDocumentationURL(std::string(sofa::SOFA_DOCUMENTATION_URL) + std::string("TODO"))
        .addDescription(R"(TODO)"));
}

} // namespace sofa
