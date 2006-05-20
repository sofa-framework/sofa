#include "Sofa/Components/EulerSolver.h"
#include "Sofa/Core/MechanicalGroup.h"
#include "Sofa/Core/MultiVector.h"
#include "XML/SolverNode.h"

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
    MultiVector pos(group, VecId::position());
    MultiVector vel(group, VecId::velocity());
    MultiVector acc(group, VecId::dx());

    if( getDebug() )
    {
        cerr<<"EulerSolver, dt = "<< dt <<endl;
        cerr<<"EulerSolver, initial x = "<< pos <<endl;
        cerr<<"EulerSolver, initial v = "<< vel <<endl;
    }

    group->computeAcc (acc, pos, vel);
    vel.peq(acc,dt);
    pos.peq(vel,dt);

    if( getDebug() )
    {
        cerr<<"EulerSolver, acceleration = "<< acc <<endl;
        cerr<<"EulerSolver, final x = "<< pos <<endl;
        cerr<<"EulerSolver, final v = "<< vel <<endl;
    }
}

void create(EulerSolver*& obj, XML::Node<Core::OdeSolver>* /*arg*/)
{
    obj = new EulerSolver();
}

SOFA_DECL_CLASS(Euler)

Creator<XML::SolverNode::Factory, EulerSolver> EulerSolverClass("Euler");

} // namespace Components

} // namespace Sofa

