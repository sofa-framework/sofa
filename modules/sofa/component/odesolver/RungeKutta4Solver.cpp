/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/odesolver/RungeKutta4Solver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>


namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace core::behavior;
using namespace sofa::defaulttype;

int RungeKutta4SolverClass = core::RegisterObject("A popular explicit time integrator")
        .add< RungeKutta4Solver >()
        .addAlias("RungeKutta4")
        ;

SOFA_DECL_CLASS(RungeKutta4);



void RungeKutta4Solver::solve (double dt, sofa::core::behavior::BaseMechanicalState::VecId xResult, sofa::core::behavior::BaseMechanicalState::VecId vResult)
{
    //sout << "RK4 Init"<<sendl;
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());
    MultiVector k1a(this, VecId::V_DERIV);
    MultiVector k2a(this, VecId::V_DERIV);
    MultiVector k3a(this, VecId::V_DERIV);
    MultiVector k4a(this, VecId::V_DERIV);
    MultiVector& k1v = vel; //(this, VecId::V_DERIV);
    MultiVector k2v(this, VecId::V_DERIV);
    MultiVector k3v(this, VecId::V_DERIV);
    MultiVector k4v(this, VecId::V_DERIV);

    MultiVector newX(this, VecId::V_COORD);
    //MultiVector newV(this, VecId::V_DERIV);

    //std::cout << "\nEntering RungeKutta4Solver::solve()\n";
    //std::cout << pos << std::endl;

    const bool log = this->f_printLog.getValue();

    double stepBy2 = double(dt / 2.0);
    double stepBy3 = double(dt / 3.0);
    double stepBy6 = double(dt / 6.0);

    double startTime = this->getTime();

    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.

    //First step
    if (log) sout << "RK4 Step 1"<<sendl;
    //k1v = vel;
    computeAcc (startTime, k1a, pos, vel);

    //Step 2
    if (log) sout << "RK4 Step 2"<<sendl;
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
        ops[0].first = (VecId)newX;
        ops[0].second.push_back(std::make_pair((VecId)pos,1.0));
        ops[0].second.push_back(std::make_pair((VecId)k1v,stepBy2));
        ops[1].first = (VecId)k2v;
        ops[1].second.push_back(std::make_pair((VecId)vel,1.0));
        ops[1].second.push_back(std::make_pair((VecId)k1a,stepBy2));
        simulation::MechanicalVMultiOpVisitor vmop(ops);
        vmop.execute(this->getContext());
    }
#endif

    computeAcc ( startTime+stepBy2, k2a, newX, k2v );

    // step 3
    if (log) sout << "RK4 Step 3"<<sendl;
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
        ops[0].first = (VecId)newX;
        ops[0].second.push_back(std::make_pair((VecId)pos,1.0));
        ops[0].second.push_back(std::make_pair((VecId)k2v,stepBy2));
        ops[1].first = (VecId)k3v;
        ops[1].second.push_back(std::make_pair((VecId)vel,1.0));
        ops[1].second.push_back(std::make_pair((VecId)k2a,stepBy2));
        simulation::MechanicalVMultiOpVisitor vmop(ops);
        vmop.execute(this->getContext());
    }
#endif

    computeAcc ( startTime+stepBy2, k3a, newX, k3v );

    // step 4
    if (log) sout << "RK4 Step 4"<<sendl;
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
        ops[0].first = (VecId)newX;
        ops[0].second.push_back(std::make_pair((VecId)pos,1.0));
        ops[0].second.push_back(std::make_pair((VecId)k3v,dt));
        ops[1].first = (VecId)k4v;
        ops[1].second.push_back(std::make_pair((VecId)vel,1.0));
        ops[1].second.push_back(std::make_pair((VecId)k3a,dt));
        simulation::MechanicalVMultiOpVisitor vmop(ops);
        vmop.execute(this->getContext());
    }
#endif

    computeAcc( startTime+dt, k4a, newX, k4v);

    MultiVector newPos(this, xResult);
    MultiVector newVel(this, vResult);

    if (log)
        sout << "RK4 Final"<<sendl;

#ifdef SOFA_NO_VMULTIOP // unoptimized version
    pos.peq(k1v,stepBy6);
    vel.peq(k1a,stepBy6);
    pos.peq(k2v,stepBy3);
    vel.peq(k2a,stepBy3);
    pos.peq(k3v,stepBy3);
    vel.peq(k3a,stepBy3);
    pos.peq(k4v,stepBy6);
    solveConstraint(dt, xResult, core::behavior::BaseConstraintSet::POS);
    vel.peq(k4a,stepBy6);
    solveConstraint(dt, vResult, core::behavior::BaseConstraintSet::VEL);
#else // single-operation optimization
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        ops[0].first = (VecId)newPos;
        ops[0].second.push_back(std::make_pair((VecId)pos,1.0));
        ops[0].second.push_back(std::make_pair((VecId)k1v,stepBy6));
        ops[0].second.push_back(std::make_pair((VecId)k2v,stepBy3));
        ops[0].second.push_back(std::make_pair((VecId)k3v,stepBy3));
        ops[0].second.push_back(std::make_pair((VecId)k4v,stepBy6));
        ops[1].first = (VecId)newVel;
        ops[1].second.push_back(std::make_pair((VecId)vel,1.0));
        ops[1].second.push_back(std::make_pair((VecId)k1a,stepBy6));
        ops[1].second.push_back(std::make_pair((VecId)k2a,stepBy3));
        ops[1].second.push_back(std::make_pair((VecId)k3a,stepBy3));
        ops[1].second.push_back(std::make_pair((VecId)k4a,stepBy6));
        simulation::MechanicalVMultiOpVisitor vmop(ops);
        vmop.execute(this->getContext());

        solveConstraint(dt, pos, core::behavior::BaseConstraintSet::POS);
        solveConstraint(dt, vel, core::behavior::BaseConstraintSet::VEL);
    }
#endif

    //std::cout << "\nExiting RungeKutta4Solver::solve()\n";
    //std::cout << pos << std::endl;

    //std::cout << "\nExiting RungeKutta4Solver::solve() : New Free pos\n";
    //std::cout << newPos << std::endl;

    simulation::MechanicalSetPositionAndVelocityVisitor spav(0, VecId::position(), VecId::velocity());
    spav.execute(this->getContext());
}



void RungeKutta4Solver::solve(double dt)
{
    //sout << "RK4 Init"<<sendl;
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());
    MultiVector k1a(this, VecId::V_DERIV);
    MultiVector k2a(this, VecId::V_DERIV);
    MultiVector k3a(this, VecId::V_DERIV);
    MultiVector k4a(this, VecId::V_DERIV);
    MultiVector& k1v = vel; //(this, VecId::V_DERIV);
    MultiVector k2v(this, VecId::V_DERIV);
    MultiVector k3v(this, VecId::V_DERIV);
    MultiVector k4v(this, VecId::V_DERIV);
    MultiVector newX(this, VecId::V_COORD);
    //MultiVector newV(this, VecId::V_DERIV);
    const bool log = this->f_printLog.getValue();

    double stepBy2 = double(dt / 2.0);
    double stepBy3 = double(dt / 3.0);
    double stepBy6 = double(dt / 6.0);

    double startTime = this->getTime();


    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.

    //First step
    if (log) sout << "RK4 Step 1"<<sendl;
    //k1v = vel;
    computeAcc (startTime, k1a, pos, vel);

    //Step 2
    if (log) sout << "RK4 Step 2"<<sendl;
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
        ops[0].first = (VecId)newX;
        ops[0].second.push_back(std::make_pair((VecId)pos,1.0));
        ops[0].second.push_back(std::make_pair((VecId)k1v,stepBy2));
        ops[1].first = (VecId)k2v;
        ops[1].second.push_back(std::make_pair((VecId)vel,1.0));
        ops[1].second.push_back(std::make_pair((VecId)k1a,stepBy2));
        simulation::MechanicalVMultiOpVisitor vmop(ops);
        vmop.execute(this->getContext());
    }
#endif

    computeAcc ( startTime+stepBy2, k2a, newX, k2v );

    // step 3
    if (log) sout << "RK4 Step 3"<<sendl;
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
        ops[0].first = (VecId)newX;
        ops[0].second.push_back(std::make_pair((VecId)pos,1.0));
        ops[0].second.push_back(std::make_pair((VecId)k2v,stepBy2));
        ops[1].first = (VecId)k3v;
        ops[1].second.push_back(std::make_pair((VecId)vel,1.0));
        ops[1].second.push_back(std::make_pair((VecId)k2a,stepBy2));
        simulation::MechanicalVMultiOpVisitor vmop(ops);
        vmop.execute(this->getContext());
    }
#endif

    computeAcc ( startTime+stepBy2, k3a, newX, k3v );

    // step 4
    if (log) sout << "RK4 Step 4"<<sendl;
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
        ops[0].first = (VecId)newX;
        ops[0].second.push_back(std::make_pair((VecId)pos,1.0));
        ops[0].second.push_back(std::make_pair((VecId)k3v,dt));
        ops[1].first = (VecId)k4v;
        ops[1].second.push_back(std::make_pair((VecId)vel,1.0));
        ops[1].second.push_back(std::make_pair((VecId)k3a,dt));
        simulation::MechanicalVMultiOpVisitor vmop(ops);
        vmop.execute(this->getContext());
    }
#endif

    computeAcc( startTime+dt, k4a, newX, k4v);

    if (log) sout << "RK4 Final"<<sendl;
#ifdef SOFA_NO_VMULTIOP // unoptimized version
    pos.peq(k1v,stepBy6);
    vel.peq(k1a,stepBy6);
    pos.peq(k2v,stepBy3);
    vel.peq(k2a,stepBy3);
    pos.peq(k3v,stepBy3);
    vel.peq(k3a,stepBy3);
    pos.peq(k4v,stepBy6);
    solveConstraint(dt,pos,core::behavior::BaseConstraintSet::POS);
    vel.peq(k4a,stepBy6);
    solveConstraint(dt,vel,core::behavior::BaseConstraintSet::VEL);
#else // single-operation optimization
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;
        VMultiOp ops;
        ops.resize(2);
        ops[0].first = (VecId)pos;
        ops[0].second.push_back(std::make_pair((VecId)pos,1.0));
        ops[0].second.push_back(std::make_pair((VecId)k1v,stepBy6));
        ops[0].second.push_back(std::make_pair((VecId)k2v,stepBy3));
        ops[0].second.push_back(std::make_pair((VecId)k3v,stepBy3));
        ops[0].second.push_back(std::make_pair((VecId)k4v,stepBy6));
        ops[1].first = (VecId)vel;
        ops[1].second.push_back(std::make_pair((VecId)vel,1.0));
        ops[1].second.push_back(std::make_pair((VecId)k1a,stepBy6));
        ops[1].second.push_back(std::make_pair((VecId)k2a,stepBy3));
        ops[1].second.push_back(std::make_pair((VecId)k3a,stepBy3));
        ops[1].second.push_back(std::make_pair((VecId)k4a,stepBy6));
        simulation::MechanicalVMultiOpVisitor vmop(ops);
        vmop.execute(this->getContext());

        solveConstraint(dt,vel,core::behavior::BaseConstraintSet::VEL);
        solveConstraint(dt,pos,core::behavior::BaseConstraintSet::POS);
    }
#endif

}



} // namespace odesolver

} // namespace component

} // namespace sofa

