#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/simulation/tree/MechanicalVisitor.h>
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
    : symplectic( initData( &symplectic, true, "symplectic", "If true, the velocities are updated before the velocities and the method is symplectic (more robust). If false, the positions are updated before the velocities (standard Euler, less robust).") )
{}

void EulerSolver::solve(double dt)
{
    MultiVector pos(this, VecId::position());
    MultiVector vel(this, VecId::velocity());
    MultiVector acc(this, VecId::dx());
    MultiVector f(this, VecId::force());
    bool printLog = f_printLog.getValue();

    if( printLog )
    {
        cerr<<"EulerSolver, dt = "<< dt <<endl;
        cerr<<"EulerSolver, initial x = "<< pos <<endl;
        cerr<<"EulerSolver, initial v = "<< vel <<endl;
    }

    addSeparateGravity(dt);	// v += dt*g . Used if mass wants to added G separately from the other forces to v.
    computeForce(f);
    accFromF(acc, f);
    projectResponse(acc);

    // update state
#ifdef SOFA_NO_VMULTIOP // unoptimized version
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
#else // single-operation optimization
    {
        simulation::tree::MechanicalVMultiOpVisitor vmop;
        vmop.ops.resize(2);
        // change order of operations depending on the sympletic flag
        int op_vel = (symplectic.getValue()?0:1);
        int op_pos = (symplectic.getValue()?1:0);
        vmop.ops[op_vel].first = (VecId)vel;
        vmop.ops[op_vel].second.push_back(std::make_pair((VecId)vel,1.0));
        vmop.ops[op_vel].second.push_back(std::make_pair((VecId)acc,dt));
        vmop.ops[op_pos].first = (VecId)pos;
        vmop.ops[op_pos].second.push_back(std::make_pair((VecId)pos,1.0));
        vmop.ops[op_pos].second.push_back(std::make_pair((VecId)vel,dt));
        vmop.execute(this->getContext());
    }
#endif

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

