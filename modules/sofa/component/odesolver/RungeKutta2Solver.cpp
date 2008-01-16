#include <sofa/component/odesolver/RungeKutta2Solver.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>
#include <iostream>
using std::cerr;
using std::endl;


namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace core::componentmodel::behavior;
using namespace sofa::defaulttype;

int RungeKutta2SolverClass = core::RegisterObject("A popular explicit time integrator")
        .add< RungeKutta2Solver >()
        .addAlias("RungeKutta2")
        ;

SOFA_DECL_CLASS(RungeKutta2);


void RungeKutta2Solver::solve(double dt)
{
    // Get the Ids of the state vectors
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());

    // Allocate auxiliary vectors
    MultiVector acc(this, VecId::V_DERIV);
    MultiVector newX(this, VecId::V_COORD);
    MultiVector newV(this, VecId::V_DERIV);

    double startTime = this->getTime();

    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.

    // Compute state derivative. vel is the derivative of pos
    computeAcc (startTime, acc, pos, vel); // acc is the derivative of vel

    // Perform a dt/2 step along the derivative
    newX = pos;
    newX.peq(vel, dt/2.); // newX = pos + vel dt/2
    newV = vel;
    newV.peq(acc, dt/2.); // newV = vel + acc dt/2

    // Compute the derivative at newX, newV
    computeAcc ( startTime+dt/2., acc, newX, newV);

    // Use the derivative at newX, newV to update the state
    pos.peq(newV,dt);
    vel.peq(acc,dt);
}



} // namespace odesolver

} // namespace component

} // namespace sofa

