#include <sofa/component/odesolver/EulerSolver.h>
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

using namespace sofa::defaulttype;
using namespace core::componentmodel::behavior;

int EulerSolverClass = core::RegisterObject("A simple explicit time integrator")
        .add< EulerSolver >()
        .addAlias("Euler")
        ;

SOFA_DECL_CLASS(Euler);

EulerSolver::EulerSolver()
    : symplectic( dataField( &symplectic, true, "symplectic", "If true, the velocities are updated before the velocities and the method is symplectic (more robust). If false, the positions are updated before the velocities (standard Euler, less robust).") )
{}

void EulerSolver::solve(double dt)
{
    //objectmodel::BaseContext* group = getContext();
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

    computeAcc ( getTime(), acc, pos, vel);

    // update state
    if( symplectic.getValue() )
    {
        vel.peq(acc,dt);
        pos.peq(vel,dt);
    }
    else
    {
        pos.peq(vel,dt);
        vel.peq(acc,dt);
    }

    if( printLog )
    {
        cerr<<"EulerSolver, acceleration = "<< acc <<endl;
        cerr<<"EulerSolver, final x = "<< pos <<endl;
        cerr<<"EulerSolver, final v = "<< vel <<endl;
    }
}

} // namespace odesolver

} // namespace component

} // namespace sofa

