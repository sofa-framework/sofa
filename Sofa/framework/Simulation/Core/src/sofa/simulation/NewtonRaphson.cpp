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
#include <sofa/simulation/DifferentialOperations.h>
#include <sofa/simulation/UpdateInternalDataVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/events/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalComputeGeometricStiffness.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalResetForceVisitor.h>



namespace sofa::simulation
{
NewtonRaphson::NewtonRaphson()
: d_maxNbIterationsNewton(initData(&d_maxNbIterationsNewton, static_cast<unsigned int>(10) , "maxNbIterationsNewton", "Maximum number of iteration for the Newton algorithm"))
, d_maxNbIterationsLineSearch(initData(&d_maxNbIterationsLineSearch, static_cast<unsigned int>(1), "maxNbIterationsLineSearch", "Maximum number of iteration for the backtracking linesearch algorithm"))
, d_newtonStepSize(initData(&d_newtonStepSize, 1.0_sreal , "newtonStepSize", "Size of the first newton step before the linesearch"))
, d_lineSearchReductionRate(initData(&d_lineSearchReductionRate, 0.5_sreal , "lineSearchReductionRate", "Taken in [0,1[ representing the fraction of diminution of the step done in the backtracking line search (if set to 0.3, the first line search will reduce the step from 1.0 to 0.7)"))
, d_lineSearchArmijoFactor(initData(&d_lineSearchArmijoFactor, 1e-3_sreal , "lineSearchArmijoFactor", "Taken in [0,1[ it represents a tolerance on the residual in term of the linear approximation. e.g., for a value of 0.01, it means we want the solution to decrease the residual as much as 0.01 times the linear approximation in the same direction."))
, d_residualThreshold(initData(&d_residualThreshold, 1e-9_sreal , "residualThreshold", "Threshold under which, the residual is considered to be sufficiently low. Newton algorithm will stop after reaching a lower value"))
, d_currentResidual(initData(&d_currentResidual , "currentResidual", "Current value of the residual"))
, d_alwaysAdvanceTime(initData(&d_alwaysAdvanceTime , false, "alwaysAdvanceTime", "Even if the residual increased after the newton step, advance time anyway."))
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
    SOFA_UNUSED(mop);

    m_mappingGraph.build(this->getContext());

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


void NewtonRaphson::newtonSolve(const core::ExecParams* params, core::ConstraintParams* cparams, common::VectorOperations* vop, common::MechanicalOperations* mop,  SReal dt, const sofa::core::MultiVecCoordId& xId, const sofa::core::MultiVecDerivId& vId, Node* node)
{
    dmsg_info() << "beginVisitor performed - SolveVisitor for freeMotion is called" ;

    //Constify the data values
    const unsigned maxNewtonIt = d_maxNbIterationsNewton.getValue();
    const unsigned maxLineSearchIt = d_maxNbIterationsLineSearch.getValue();
    const SReal newtonStepSize = d_newtonStepSize.getValue();
    const SReal residualThreshold = d_residualThreshold.getValue();
    const SReal lineSearchReductionRate = d_lineSearchReductionRate.getValue();
    const SReal lineSearchArmijoFactor = d_lineSearchArmijoFactor.getValue();

    const auto pos = sofa::core::vec_id::write_access::position;
    const auto vel = sofa::core::vec_id::write_access::velocity;

    const bool printLog = f_printLog.getValue();

    m_advanceTime = true;

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

    SReal initialResidual = intHelper.evaluateResidual(false);
    SReal oldResidual = initialResidual;
    SReal newResidual = oldResidual;


    unsigned it = 0;
    while ( it<maxNewtonIt && newResidual>residualThreshold )
    {
        SCOPED_TIMER_VARNAME(step_timer, "NewtonStep");


        //TODO do we want to recompute this each newton step ?
        // computeGeometricalStiffness(&mop->mparams, cparams, node);

        //Upate old residual to be the one of the last newton iteration
        oldResidual = newResidual;

        double bestresidual = oldResidual;
        double bestalpha = 0.0;

        if (it != 0)
        {
            intHelper.computeRHS(false);
        }
        intHelper.computeLHS();
        //Find decrease direction
        intHelper.solveLinearEquation();

        //Setup variables for linesearch
        SReal alpha = newtonStepSize;
        SReal delta = 0.0;

        //Already make a full step
        intHelper.updateStatesFromLinearSolution(alpha);
        newResidual = intHelper.evaluateResidual(true);

        // Compute the Armijo term for the cost function f = ||r||^2.
        // With J = dr/dX, we have \nabla f = 2 * J^T * r, hence \nabla f . p = 2 * r^T * J * p.
        // The Newton direction satisfies J * p = -r, so \nabla f . p = -2 * ||r||^2.
        // This holds for any (even non-symmetric) J, but it requires the system matrix A to be the
        // exact Jacobian of the residual. If A is only an approximation of J (e.g. missing
        // geometric stiffness, or interaction forcefields contributing only to the RHS), this
        // term is an approximation and the Armijo condition may be too strict or too loose.
        const SReal armijoTerm = - 2 * lineSearchArmijoFactor * oldResidual;

        if (newResidual<bestresidual)
        {
            bestresidual = newResidual;
            bestalpha = alpha;
        }

        unsigned lineSearchIt = 0;
        while ((newResidual>(oldResidual + alpha*armijoTerm)) && lineSearchIt<maxLineSearchIt )
        {
            //We are backtracking on the same line. Instead of starting from initial position and
            //adding alpha each time, we go back toward the initial position
            delta = alpha*lineSearchReductionRate;
            alpha -= delta;

            intHelper.updateStatesFromLinearSolution(-delta);
            newResidual = intHelper.evaluateResidual(true);

            if (newResidual<bestresidual)
            {
                bestresidual = newResidual;
                bestalpha = alpha;
            }

            ++lineSearchIt;
        }

        if (fabs(alpha - bestalpha )> std::numeric_limits<SReal>::epsilon() )
        {
            intHelper.updateStatesFromLinearSolution( bestalpha - alpha);
            newResidual = intHelper.evaluateResidual(true);
        }

        msg_info_when(printLog)<<"| ---------- Newton step = "<<it + 1<<" ---------- |";
        msg_info_when(printLog)<<"Number of line search iterations = "<<lineSearchIt;

        if (bestalpha < std::numeric_limits<SReal>::epsilon())
        {

            msg_warning_when(printLog)<<"Line-search didn't improve anything ! Making the full newton step...";

            //LineSearch didn't improve anything let's advance the newton.
            intHelper.updateStatesFromLinearSolution( 1.0 - bestalpha);
            newResidual = intHelper.evaluateResidual(true);
            bestalpha = 1.0;
        }

        msg_info_when(printLog)<<"Best alpha : "<<bestalpha;
        msg_info_when(printLog)<<"Residual evolution : "<< oldResidual << " --> "<<newResidual ;
        if (newResidual>oldResidual)
        {
            msg_warning_when(printLog)<<"NewtonStep increased the residual !";
        }
        msg_info_when(printLog)<<"| ------------------------------------- |";

        ++it ;
    }

    // If the newton didn't converge to a better solution, don't advance
    if (newResidual>initialResidual && !d_alwaysAdvanceTime.getValue())
    {
        m_advanceTime=false;
    }

    if (printLog)
    {
        if (newResidual<residualThreshold)
        {
            msg_info()<<"Newton converged to residual "<<newResidual<<" in "<<it<<" steps.";
        }
        else
        {
            msg_warning()<<"Newton didn't converge ! Current residual is "<<newResidual;
        }
    }

    helper::AdvancedTimer::valSet("nb_iterations", it);
    helper::AdvancedTimer::valSet("residual", std::sqrt(newResidual));

    d_currentResidual.setValue(newResidual);

    if (m_advanceTime)
    {
        //Integrate explicit IS
        intHelper.integrateExplicitIS();
        if (m_solveConstraints)
        {
            solveConstraints(&mop->mparams, cparams, vop, mop, pos, vel,dt);
        }
    }
    intHelper.finalizeIntegrationStep(m_advanceTime);

    mechanicalvisitor::MechanicalResetForceVisitor resetForceVisitor(params, sofa::core::vec_id::write_access::externalForce);
    node->execute(&resetForceVisitor);

}

void NewtonRaphson::postSolve(const core::ExecParams* params, core::ConstraintParams* cparams, common::VectorOperations* vop, common::MechanicalOperations* mop, SReal startTime, SReal dt, Node* node)
{
    SOFA_UNUSED(cparams);
    SOFA_UNUSED(vop);

    auto pos = sofa::core::vec_id::write_access::position;
    auto vel = sofa::core::vec_id::write_access::velocity;

    if (cparams->constOrder() == core::ConstraintOrder::POS)
    {
        SCOPED_TIMER("ProjectAndPropagateX");

        mop->projectPosition(pos);

        simulation::common::DifferentialOperations::pushforwardCoord(m_mappingGraph, mop->mparams, pos);
    }
    else
    {
        SCOPED_TIMER("ProjectAndPropagateXAndV");

        mop->projectPositionAndVelocity(pos, vel);

        simulation::common::DifferentialOperations::pushforwardCoord(m_mappingGraph, mop->mparams, pos);
        simulation::common::DifferentialOperations::pushforwardTangent(m_mappingGraph, mop->mparams, vel);
    }

    if (m_advanceTime)
    {
        node->setTime ( startTime + dt );
        node->execute<UpdateSimulationContextVisitor>(params);  // propagate time
    }

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
    SOFA_UNUSED(mparams);
    SCOPED_TIMER_VARNAME(ConstraintSolverTimer, "ConstraintSolver");

    const auto constOrder = cparams->constOrder();
    if (constOrder == core::ConstraintOrder::VEL )
    {
        l_constraintSolver->solveConstraint(cparams, vId);
        //TODO Why is this even necessary, shouldn't we call the integraitons cheme to perform this integration ?
        vop->v_peq(xId, vId, dt); //position += velocity * dt
    }
    else if (constOrder == core::ConstraintOrder::POS_AND_VEL)
    {
        l_constraintSolver->solveConstraint(cparams, xId, vId);
    }
    else if (constOrder == core::ConstraintOrder::POS)
    {
        l_constraintSolver->solveConstraint(cparams, xId);
    }

    {
        SCOPED_TIMER_VARNAME(ProjectAndPropagateDxTimer, "ProjectAndPropagateDx");

        core::behavior::MultiVecDeriv cdx(vop, l_constraintSolver->getDx());
        mop->projectResponse(cdx);
        simulation::common::DifferentialOperations::pushforwardTangent(m_mappingGraph, mop->mparams, cdx);
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
