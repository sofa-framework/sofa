#include "Sofa/Components/RungeKutta4Solver.h"
#include "Sofa/Core/MultiVector.h"
#include "Common/ObjectFactory.h"

#include <math.h>

namespace Sofa
{

namespace Components
{

using namespace Core;
using namespace Common;

void RungeKutta4Solver::solve(double dt)
{
    MultiVector pos(group, VecId::position());
    MultiVector vel(group, VecId::velocity());
    MultiVector k1a(group, V_DERIV);
    MultiVector k2a(group, V_DERIV);
    MultiVector k3a(group, V_DERIV);
    MultiVector k4a(group, V_DERIV);
    MultiVector k1v(group, V_DERIV);
    MultiVector k2v(group, V_DERIV);
    MultiVector k3v(group, V_DERIV);
    MultiVector k4v(group, V_DERIV);
    MultiVector newX(group, V_COORD);
    MultiVector newV(group, V_DERIV);

    double stepBy2 = double(dt / 2.0);
    double stepBy3 = double(dt / 3.0);
    double stepBy6 = double(dt / 6.0);

    //First step
    group->computeAcc (k1a, pos, vel);
    k1v = vel;

    //Step 2
    newX = pos;
    newV = vel;

    newX.peq(k1v, stepBy2);
    newV.peq(k1a, stepBy2);

    k2v = newV;
    group->computeAcc (k2a, newX, newV);

    // step 3
    newX = pos;
    newV = vel;

    newX.peq(k2v, stepBy2);
    newV.peq(k2a, stepBy2);

    k3v = newV;
    group->computeAcc (k3a, newX, newV);

    // step 4
    newX = pos;
    newV = vel;
    newX.peq(k3v, dt);
    newV.peq(k3a, dt);

    k4v = newV;
    group->computeAcc(k4a, newX, newV);

    pos.peq(k1v,stepBy6);
    vel.peq(k1a,stepBy6);
    pos.peq(k2v,stepBy3);
    vel.peq(k2a,stepBy3);
    pos.peq(k3v,stepBy3);
    vel.peq(k3a,stepBy3);
    pos.peq(k4v,stepBy6);
    vel.peq(k4a,stepBy6);
}

void create(RungeKutta4Solver*& obj, ObjectDescription* /*arg*/)
{
    obj = new RungeKutta4Solver();
}

SOFA_DECL_CLASS(RungeKutta4)

Creator<ObjectFactory, RungeKutta4Solver> RungeKutta4SolverClass("RungeKutta4");

} // namespace Components

} // namespace Sofa
