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
#include <sofa/component/odesolver/backward/StaticOdeSolver.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>

namespace sofa::component::odesolver::backward
{

void registerStaticOdeSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(
        core::ObjectRegistrationData("Static ODE Solver")
            .add<StaticOdeSolver>());
}

StaticOdeSolver::StaticOdeSolver()
    : l_newtonSolver(initLink("newtonSolver", "Link to a NewtonRaphsonSolver"))
{}

void StaticOdeSolver::init()
{
    OdeSolver::init();
    LinearSolverAccessor::init();

    if (!l_newtonSolver.get())
    {
        l_newtonSolver.set(getContext()->get<NewtonRaphsonSolver>(getContext()->getTags(), core::objectmodel::BaseContext::SearchDown));

        if (!l_newtonSolver)
        {
            msg_error() << "A Newton-Raphson solver is required by this component but has not been found.";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        }
    }

    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Invalid)
    {
        d_componentState.setValue(core::objectmodel::ComponentState::Valid);
    }
}

struct StaticResidualFunction : newton_raphson::BaseNonLinearFunction
{
    void evaluateCurrentGuess() override
    {
        SCOPED_TIMER("ComputeForce");
        static constexpr bool clearForcesBeforeComputingThem = true;
        static constexpr bool applyBottomUpMappings = true;

        mop.computeForce(force, clearForcesBeforeComputingThem, applyBottomUpMappings);
        mop.projectResponse(force);
    }

    SReal squaredNormLastEvaluation() override
    {
        return force.dot(force);
    }

    void computeGradientFromCurrentGuess() override
    {
        SCOPED_TIMER("ComputeGradient");

        static constexpr core::MatricesFactors::M m(0);
        static constexpr core::MatricesFactors::B b(0);
        static constexpr core::MatricesFactors::K k(-1);

        mop.setSystemMBKMatrix(m, b, k, linearSolver);
    }

    void updateGuessFromLinearSolution(SReal alpha) override
    {
        x.peq(dx, alpha);
        mop.solveConstraint(x, sofa::core::ConstraintOrder::POS);
        mop.propagateX(x);
    }

    void solveLinearEquation() override
    {
        linearSolver->setSystemLHVector(dx);
        linearSolver->setSystemRHVector(force);
        linearSolver->solveSystem();
    }

    SReal squaredNormDx() override
    {
        return dx.dot(dx);
    }

    SReal squaredLastEvaluation() override
    {
        return x.dot(x);
    }

    sofa::simulation::common::MechanicalOperations& mop;
    core::behavior::MultiVecCoord& x;
    core::behavior::MultiVecDeriv& force;
    core::behavior::MultiVecDeriv& dx;
    core::behavior::LinearSolver* linearSolver { nullptr };

    StaticResidualFunction(sofa::simulation::common::MechanicalOperations& mop,
                           core::behavior::MultiVecCoord& x, core::behavior::MultiVecDeriv& force,
                           core::behavior::MultiVecDeriv& dx,
                           core::behavior::LinearSolver* linearSolver)
        : mop(mop), x(x), force(force), dx(dx), linearSolver(linearSolver)
    {
    }
};

void StaticOdeSolver::solve(const core::ExecParams* params, SReal dt, core::MultiVecCoordId xResult,
                            core::MultiVecDerivId vResult)
{
    if (!isComponentStateValid())
    {
        return;
    }

    // Create the vector and mechanical operations tools. These are used to execute special
    // operations (multiplication,
    // additions, etc.) on multi-vectors (a vector that is stored in different buffers inside the
    // mechanical objects)
    sofa::simulation::common::VectorOperations vop(params, this->getContext());
    sofa::simulation::common::MechanicalOperations mop(params, this->getContext());
    mop->setImplicit(true);

    core::behavior::MultiVecCoord x(&vop, xResult );

    core::behavior::MultiVecDeriv force(&vop, sofa::core::vec_id::write_access::force);

    core::behavior::MultiVecDeriv dx(&vop, core::vec_id::write_access::dx);
    dx.realloc(&vop, true, true);

    StaticResidualFunction staticResidualFunction(mop, x, force, dx, l_linearSolver.get());
    l_newtonSolver->solve(staticResidualFunction);
}


}  // namespace sofa::component::odesolver::backward
