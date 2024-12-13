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
#include <sofa/component/odesolver/backward/NewtonRaphsonSolver.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>


namespace sofa::component::odesolver::backward
{

NewtonRaphsonSolver::NewtonRaphsonSolver()
    : l_integrationMethod(initLink("integrationMethod", "The integration method to use in a Newton iteration"))
{}

void NewtonRaphsonSolver::init()
{
    OdeSolver::init();
    LinearSolverAccessor::init();

    if (!l_integrationMethod.get())
    {
        l_integrationMethod.set(getContext()->get<core::behavior::BaseIntegrationMethod>(getContext()->getTags(), core::objectmodel::BaseContext::SearchDown));

        if (!l_integrationMethod)
        {
            msg_error() << "An integration method is required by this component but has not been found.";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
    }

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void NewtonRaphsonSolver::solve(
    const core::ExecParams* params, SReal dt,
    sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    if (!isComponentStateValid())
    {
        return;
    }

    // Create the vector and mechanical operations tools. These are used to execute special operations (multiplication,
    // additions, etc.) on multi-vectors (a vector that is stored in different buffers inside the mechanical objects)
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    mop->setImplicit(true);

    //current position computed at the end of the previous time step
    core::behavior::MultiVecCoord position(&vop, core::vec_id::write_access::position );

    //current position computed at the end of the previous time step
    core::behavior::MultiVecDeriv velocity(&vop, core::vec_id::write_access::velocity );

    //force vector that will be computed in this solve. The content of the
    //previous time step will be erased.
    core::behavior::MultiVecDeriv force(&vop, core::vec_id::write_access::force );
    core::behavior::MultiVecDeriv b(&vop, true, core::VecIdProperties{"RHS", GetClass()->className});

    //the intermediate position and velocity required by the algorithm
    core::behavior::MultiVecCoord position_i(&vop, xResult );
    core::behavior::MultiVecDeriv velocity_i(&vop, vResult );

    //the position and velocity computed at the end of this algorithm
    core::behavior::MultiVecCoord newPosition(&vop, xResult );
    core::behavior::MultiVecDeriv newVelocity(&vop, vResult );

    //dx vector is required by some operations of the algorithm, even if it is
    //not explicit
    core::behavior::MultiVecDeriv dx(&vop, sofa::core::vec_id::write_access::dx);
    dx.realloc(&vop, false, true);

    // inform the constraint parameters about the position and velocity id
    mop.cparams.setX(xResult);
    mop.cparams.setV(vResult);

    l_integrationMethod->initializeVectors(position, velocity);

    {
        SCOPED_TIMER("ComputeRHS");

        core::behavior::RHSInput input;
        input.force = force;

        l_integrationMethod->computeRightHandSide(params, input, b, dt);
    }

    SReal error{};
    {
        SCOPED_TIMER("ComputeError");
        error = b.dot(b);
    }


}

void registerNewtonRaphsonSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Newton-Raphson algorithm.")
        .add< NewtonRaphsonSolver >());
}

}
