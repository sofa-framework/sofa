#include "Sofa/Components/EulerSolver.h"
#include "Sofa/Core/MechanicalGroup.h"
#include "Sofa/Core/MultiVector.h"
#include "XML/SolverNode.h"

#include <math.h>

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

    group->computeAcc (acc, pos, vel);
    vel.peq(acc,dt);
    pos.peq(vel,dt);
}

void create(EulerSolver*& obj, XML::Node<Core::OdeSolver>* /*arg*/)
{
    obj = new EulerSolver();
}

Creator<XML::SolverNode::Factory, EulerSolver> EulerSolverClass("Euler");

} // namespace Components

} // namespace Sofa
