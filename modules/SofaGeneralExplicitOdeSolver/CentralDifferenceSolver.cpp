/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaGeneralExplicitOdeSolver/CentralDifferenceSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>
#include <sofa/helper/system/thread/CTime.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

using core::VecId;
using namespace sofa::defaulttype;
using namespace core::behavior;

CentralDifferenceSolver::CentralDifferenceSolver()
    : f_rayleighMass( initData(&f_rayleighMass,(SReal)0.1,"rayleighMass","Rayleigh damping coefficient related to mass"))
{
}

/**
 *
 *  Given current and last position $u_n$ and $u_{n-1}$, the next position $u_{n+1}$ is computed as follow:
 *
 *  Velocity: $\dot{u}_{n-1/2} = \frac{u_n - u_{n-1}}{dt}$ $\dot{u}_{n+1/2} = \frac{u_{n+1} - u_n}{dt}$ $\dot{u}_n = \frac{u_{n+1} - u_{n-1}}{2dt}$
 *
 *  Acceleration: $\ddot{u}_n = \frac{\dot{u}_{n+1/2} - \dot{u}_{n-1/2}}{dt}$
 *
 *  Equilibrium: $M \ddot{u}_n + C \dot{u}_n + K u_n = P_n$
 *
 *  Solving for $u_{n+1}$, given $u_n$ and $u_{n-1}$:
 *
 *  $\ddot{u}_n = \frac{u_{n+1} - 2u_n + u_{n-1}}{dt^2}$
 *
 *  $( M + \frac{dt}{2} C ) u_{n+1} = dt^2 P_n - (dt^2 K - 2M) u_n - ( M - \frac{dt}{2} C ) u_{n-1}$
 *  $ M u_{n+1} = dt^2 P_n - dt^2 K u_n + 2 M u_n - M u_{n-1}$
 *
 *  Solving for $\dot{u}_{n+1/2}$, given $u_n$ and $u_{n-1/2}$:
 *
 *  $\dot{u}_n = \frac{\dot{u}_{n+1/2} + \dot{u}_{n-1/2}}{2}$
 *
 *  $M \frac{\dot{u}_{n+1/2} - \dot{u}_{n-1/2}}{dt} + C \frac{\dot{u}_{n+1/2} + \dot{u}_{n-1/2}}{2} + K u_n = P_n$
 *
 *  $(\frac{M}{dt} + \frac{C}{2}) \dot{u}_{n+1/2} = P_n - K u_n - ( \frac{M}{dt} - \frac{C}{2} ) \dot{u}_{n-1/2} $
 *
 *  If using rayleigh damping: $C = rM$
 *
 *  $(\frac{1}{dt} + \frac{r}{2})M \dot{u}_{n+1/2} = P_n - K u_n - ( \frac{1}{dt} - \frac{r}{2} ) M \dot{u}_{n-1/2} $
 *
 *  $ \dot{u}_{n+1/2} = \frac{\frac{1}{dt} - \frac{r}{2}}{\frac{1}{dt} + \frac{r}{2}} \dot{u}_{n-1/2} + \frac{1}{\frac{1}{dt} + \frac{r}{2}} M^{-1} ( P_n - K u_n ) $
 *
 *  $u_{n+1} = u_n + dt \dot{u}_{n+1/2}$
 *
 */

void CentralDifferenceSolver::solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    mop->setImplicit(false); // this solver is explicit only
    MultiVecCoord pos(&vop, core::VecCoordId::position() );
    MultiVecDeriv vel(&vop, core::VecDerivId::velocity() );
    MultiVecCoord pos2(&vop, xResult /*core::VecCoordId::position()*/ );
    MultiVecDeriv vel2(&vop, vResult /*core::VecDerivId::velocity()*/ );
    MultiVecDeriv dx (&vop, core::VecDerivId::dx() ); dx.realloc( &vop, true, true );
    MultiVecDeriv f  (&vop, core::VecDerivId::force() );

    const SReal r = f_rayleighMass.getValue();

    mop.addSeparateGravity(dt);                // v += dt*g . Used if mass wants to added G separately from the other forces to v.

    //projectVelocity(vel);                  // initial velocities are projected to the constrained space

    // compute the current force
    mop.computeForce(f);                       // f = P_n - K u_n

    mop.accFromF(dx, f);                       // dx = M^{-1} ( P_n - K u_n )
    mop.projectResponse(dx);                    // dx is projected to the constrained space

    mop.solveConstraint(dx, core::ConstraintParams::ACC);
    // apply the solution
    if (r==0)
    {
#ifdef SOFA_NO_VMULTIOP // unoptimized version
        vel2.eq( vel, dx, dt );                  // vel = vel + dt M^{-1} ( P_n - K u_n )
        mop.solveConstraint(vel2, core::ConstraintParams::VEL);
        pos2.eq( pos, vel2, dt );                    // pos = pos + h vel
        mop.solveConstraint(pos2, core::ConstraintParams::POS);

#else // single-operation optimization

        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        // vel += dx * dt
        ops[0].first = vel2;
        ops[0].second.push_back(std::make_pair(vel.id(),1.0));
        ops[0].second.push_back(std::make_pair(dx.id(),dt));
        // pos += vel * dt
        ops[1].first = pos2;
        ops[1].second.push_back(std::make_pair(pos.id(),1.0));
        ops[1].second.push_back(std::make_pair(vel2.id(),dt));

        vop.v_multiop(ops);

        mop.solveConstraint(vel2,core::ConstraintParams::VEL);
        mop.solveConstraint(pos2,core::ConstraintParams::POS);
#endif
    }
    else
    {
#ifdef SOFA_NO_VMULTIOP // unoptimized version
        vel2.eq( vel, (1/dt - r/2)/(1/dt + r/2) );
        vel2.peq( dx, 1/(1/dt + r/2) );     // vel = \frac{\frac{1}{dt} - \frac{r}{2}}{\frac{1}{dt} + \frac{r}{2}} vel + \frac{1}{\frac{1}{dt} + \frac{r}{2}} M^{-1} ( P_n - K u_n )
        pos2.eq( pos, vel2, dt );                    // pos = pos + h vel
#else // single-operation optimization
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        // vel += dx * dt
        ops[0].first = vel2;
        ops[0].second.push_back(std::make_pair(vel.id(),1.0));
        ops[0].second.push_back(std::make_pair(vel.id(),-1.0 + (1/dt - r/2)/(1/dt + r/2)));
        ops[0].second.push_back(std::make_pair(dx.id(),1/(1/dt + r/2)));
        // pos += vel * dt
        ops[1].first = pos2;
        ops[1].second.push_back(std::make_pair(pos.id(),1.0));
        ops[1].second.push_back(std::make_pair(vel2.id(),dt));

        vop.v_multiop(ops);

        mop.solveConstraint(vel2,core::ConstraintParams::VEL);
        mop.solveConstraint(pos2,core::ConstraintParams::POS);
#endif
    }

}

SOFA_DECL_CLASS(CentralDifferenceSolver)

int CentralDifferenceSolverClass = core::RegisterObject("Explicit time integrator using central difference (also known as Verlet of Leap-frop)")
        .add< CentralDifferenceSolver >()
        .addAlias("CentralDifference");
;

} // namespace odesolver

} // namespace component

} // namespace sofa

