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
#include <sofa/component/odesolver/forward/RungeKutta2Solver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/core/ObjectFactory.h>




namespace sofa::component::odesolver::forward
{
using core::VecId;
using namespace core::behavior;
using namespace sofa::defaulttype;

void registerRungeKutta2Solver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A popular explicit time integrator.")
        .add< RungeKutta2Solver >());
}

void RungeKutta2Solver::solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    mop->setImplicit(false); // this solver is explicit only
    // Get the Ids of the state vectors
    MultiVecCoord pos(&vop, core::vec_id::write_access::position );
    MultiVecDeriv vel(&vop, core::vec_id::write_access::velocity );
    MultiVecCoord pos2(&vop, xResult /*core::vec_id::write_access::position*/ );
    MultiVecDeriv vel2(&vop, vResult /*core::vec_id::write_access::velocity*/ );

    // Allocate auxiliary vectors
    MultiVecDeriv acc(&vop);
    MultiVecCoord newX(&vop);
    MultiVecDeriv newV(&vop);

    SReal startTime = this->getTime();

    mop.addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.

    // Compute state derivative. vel is the derivative of pos
    mop.computeAcc (startTime, acc, pos, vel); // acc is the derivative of vel

    // Perform a dt/2 step along the derivative
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    newX = pos;
    newX.peq(vel, dt/2.); // newX = pos + vel dt/2
    newV = vel;
    newV.peq(acc, dt/2.); // newV = vel + acc dt/2
#else // single-operation optimization
    {

        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        ops[0].first = newX;
        ops[0].second.push_back(std::make_pair(pos.id(),1.0));
        ops[0].second.push_back(std::make_pair(vel.id(),dt/2));
        ops[1].first = newV;
        ops[1].second.push_back(std::make_pair(vel.id(),1.0));
        ops[1].second.push_back(std::make_pair(acc.id(),dt/2));

        vop.v_multiop(ops);
    }
#endif

    // Compute the derivative at newX, newV
    mop.computeAcc ( startTime+dt/2., acc, newX, newV);

    // Use the derivative at newX, newV to update the state
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    pos2.eq(pos,newV,dt);
    mop.solveConstraint(pos2,core::ConstraintOrder::POS);
    vel2.eq(vel,acc,dt);
    mop.solveConstraint(vel2,core::ConstraintOrder::VEL);
#else // single-operation optimization
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        ops[0].first = pos2;
        ops[0].second.push_back(std::make_pair(pos.id(),1.0));
        ops[0].second.push_back(std::make_pair(newV.id(),dt));
        ops[1].first = vel2;
        ops[1].second.push_back(std::make_pair(vel.id(),1.0));
        ops[1].second.push_back(std::make_pair(acc.id(),dt));
        vop.v_multiop(ops);

        mop.solveConstraint(vel2,core::ConstraintOrder::VEL);
        mop.solveConstraint(pos2,core::ConstraintOrder::POS);
    }
#endif


}



} // namespace sofa::component::odesolver::forward
