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
#include <sofa/component/odesolver/forward/RungeKutta4Solver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/core/ObjectFactory.h>
#include <cmath>


namespace sofa::component::odesolver::forward
{

using core::VecId;
using namespace core::behavior;
using namespace sofa::defaulttype;

void registerRungeKutta4Solver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A popular explicit time integrator.")
        .add< RungeKutta4Solver >());
}

void RungeKutta4Solver::solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
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
    MultiVecDeriv k1a(&vop);
    MultiVecDeriv k2a(&vop);
    MultiVecDeriv k3a(&vop);
    MultiVecDeriv k4a(&vop);
    MultiVecDeriv& k1v = vel; //(&vop);
    MultiVecDeriv k2v(&vop);
    MultiVecDeriv k3v(&vop);
    MultiVecDeriv k4v(&vop);

    MultiVecCoord newX(&vop);

    SReal stepBy2 = SReal(dt / 2.0);
    SReal stepBy3 = SReal(dt / 3.0);
    SReal stepBy6 = SReal(dt / 6.0);

    SReal startTime = this->getTime();

    mop.addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.

    //First step
    dmsg_info() << "RK4 Step 1";

    //k1v = vel;
    mop.computeAcc (startTime, k1a, pos, vel);

    //Step 2
    dmsg_info() << "RK4 Step 2" ;

#ifdef SOFA_NO_VMULTIOP // unoptimized version
    newX = pos;
    k2v = vel;
    newX.peq(k1v, stepBy2);
    k2v.peq(k1a, stepBy2);
#else // single-operation optimization
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        ops[0].first = newX;
        ops[0].second.push_back(std::make_pair(pos.id(),1.0));
        ops[0].second.push_back(std::make_pair(k1v.id(),stepBy2));
        ops[1].first = k2v;
        ops[1].second.push_back(std::make_pair(vel.id(),1.0));
        ops[1].second.push_back(std::make_pair(k1a.id(),stepBy2));
        vop.v_multiop(ops);
    }
#endif

    mop.computeAcc ( startTime+stepBy2, k2a, newX, k2v );

    // step 3
    dmsg_info() << "RK4 Step 3" ;
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    newX = pos;
    k3v = vel;
    newX.peq(k2v, stepBy2);
    k3v.peq(k2a, stepBy2);
#else // single-operation optimization
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        ops[0].first = newX;
        ops[0].second.push_back(std::make_pair(pos.id(),1.0));
        ops[0].second.push_back(std::make_pair(k2v.id(),stepBy2));
        ops[1].first = k3v;
        ops[1].second.push_back(std::make_pair(vel.id(),1.0));
        ops[1].second.push_back(std::make_pair(k2a.id(),stepBy2));
        vop.v_multiop(ops);
    }
#endif

    mop.computeAcc ( startTime+stepBy2, k3a, newX, k3v );

    // step 4
    dmsg_info() << "RK4 Step 4" ;
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    newX = pos;
    k4v = vel;
    newX.peq(k3v, dt);
    k4v.peq(k3a, dt);
#else // single-operation optimization
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        ops[0].first = newX;
        ops[0].second.push_back(std::make_pair(pos.id(),1.0));
        ops[0].second.push_back(std::make_pair(k3v.id(),dt));
        ops[1].first = k4v;
        ops[1].second.push_back(std::make_pair(vel.id(),1.0));
        ops[1].second.push_back(std::make_pair(k3a.id(),dt));
        vop.v_multiop(ops);
    }
#endif

    mop.computeAcc( startTime+dt, k4a, newX, k4v);

   dmsg_info() << "RK4 Final";

#ifdef SOFA_NO_VMULTIOP // unoptimized version
    pos2.eq(pos,k1v,stepBy6);
    vel2.eq(vel,k1a,stepBy6);
    pos2.peq(k2v,stepBy3);
    vel2.peq(k2a,stepBy3);
    pos2.peq(k3v,stepBy3);
    vel2.peq(k3a,stepBy3);
    pos2.peq(k4v,stepBy6);
    mop.solveConstraint(pos2, core::ConstraintOrder::POS);
    vel2.peq(k4a,stepBy6);
    mop.solveConstraint(vel2, core::ConstraintOrder::VEL);
#else // single-operation optimization
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        ops[0].first = pos2;
        ops[0].second.push_back(std::make_pair(pos.id(),1.0));
        ops[0].second.push_back(std::make_pair(k1v.id(),stepBy6));
        ops[0].second.push_back(std::make_pair(k2v.id(),stepBy3));
        ops[0].second.push_back(std::make_pair(k3v.id(),stepBy3));
        ops[0].second.push_back(std::make_pair(k4v.id(),stepBy6));
        ops[1].first = vel2;
        ops[1].second.push_back(std::make_pair(vel.id(),1.0));
        ops[1].second.push_back(std::make_pair(k1a.id(),stepBy6));
        ops[1].second.push_back(std::make_pair(k2a.id(),stepBy3));
        ops[1].second.push_back(std::make_pair(k3a.id(),stepBy3));
        ops[1].second.push_back(std::make_pair(k4a.id(),stepBy6));
        vop.v_multiop(ops);

        mop.solveConstraint(pos, core::ConstraintOrder::POS);
        mop.solveConstraint(vel, core::ConstraintOrder::VEL);
    }
#endif
}


} // namespace sofa::component::odesolver::forward
