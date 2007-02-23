#include <sofa/component/odesolver/RungeKutta4Solver.h>
#include <sofa/core/ObjectFactory.h>
#include <math.h>


namespace sofa
{

namespace component
{

namespace odesolver
{

using namespace core::componentmodel::behavior;
using namespace sofa::defaulttype;

int RungeKutta4SolverClass = core::RegisterObject("A popular explicit time integrator")
        .add< RungeKutta4Solver >()
        .addAlias("RungeKutta4")
        ;

SOFA_DECL_CLASS(RungeKutta4);


void RungeKutta4Solver::solve(double dt)
{
    //std::cout << "RK4 Init\n";
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());
    MultiVector k1a(this, VecId::V_DERIV);
    MultiVector k2a(this, VecId::V_DERIV);
    MultiVector k3a(this, VecId::V_DERIV);
    MultiVector k4a(this, VecId::V_DERIV);
    MultiVector k1v(this, VecId::V_DERIV);
    MultiVector k2v(this, VecId::V_DERIV);
    MultiVector k3v(this, VecId::V_DERIV);
    MultiVector k4v(this, VecId::V_DERIV);
    MultiVector newX(this, VecId::V_COORD);
    MultiVector newV(this, VecId::V_DERIV);

    double stepBy2 = double(dt / 2.0);
    double stepBy3 = double(dt / 3.0);
    double stepBy6 = double(dt / 6.0);

    double startTime = this->getTime();

    //First step
    //std::cout << "RK4 Step 1\n";
    k1v = vel;
    computeAcc (startTime, k1a, pos, vel);

    //Step 2
    //std::cout << "RK4 Step 2\n";
    newX = pos;
    newV = vel;

    newX.peq(k1v, stepBy2);
    newV.peq(k1a, stepBy2);

    k2v = newV;
    computeAcc ( startTime+stepBy2, k2a, newX, newV);

    // step 3
    //std::cout << "RK4 Step 3\n";
    newX = pos;
    newV = vel;

    newX.peq(k2v, stepBy2);
    newV.peq(k2a, stepBy2);

    k3v = newV;
    computeAcc ( startTime+stepBy2, k3a, newX, newV);

    // step 4
    //std::cout << "RK4 Step 4\n";
    newX = pos;
    newV = vel;
    newX.peq(k3v, dt);
    newV.peq(k3a, dt);

    k4v = newV;
    computeAcc( startTime+dt, k4a, newX, newV);

    //std::cout << "RK4 Final\n";
    pos.peq(k1v,stepBy6);
    vel.peq(k1a,stepBy6);
    pos.peq(k2v,stepBy3);
    vel.peq(k2a,stepBy3);
    pos.peq(k3v,stepBy3);
    vel.peq(k3a,stepBy3);
    pos.peq(k4v,stepBy6);
    vel.peq(k4a,stepBy6);
}



} // namespace odesolver

} // namespace component

} // namespace sofa

