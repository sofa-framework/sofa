#include "Sofa/Components/EulerSolver.h"
#include "Sofa/Core/MultiVector.h"
#include "Common/ObjectFactory.h"

#include <math.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Core;

void EulerSolver::solve(double dt)
{
    //Abstract::BaseContext* group = getContext();
    OdeSolver* group = this;
    MultiVector pos(group, VecId::position());
    MultiVector vel(group, VecId::velocity());
    MultiVector acc(group, VecId::dx());
    bool printLog = f_printLog.getValue();

    if( printLog )
    {
        cerr<<"EulerSolver, dt = "<< dt <<endl;
        cerr<<"EulerSolver, initial x = "<< pos <<endl;
        cerr<<"EulerSolver, initial v = "<< vel <<endl;
    }

    group->computeAcc ( getTime(), acc, pos, vel);
    vel.peq(acc,dt);
    pos.peq(vel,dt);

    if( printLog )
    {
        cerr<<"EulerSolver, acceleration = "<< acc <<endl;
        cerr<<"EulerSolver, final x = "<< pos <<endl;
        cerr<<"EulerSolver, final v = "<< vel <<endl;
    }
}

void create(EulerSolver*& obj, ObjectDescription* arg)
{
    obj = new EulerSolver();
    obj->parseFields( arg->getAttributeMap() );
}

SOFA_DECL_CLASS(Euler)

Creator<ObjectFactory, EulerSolver> EulerSolverClass("Euler");

} // namespace Components

} // namespace Sofa


