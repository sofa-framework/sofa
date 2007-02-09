#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>
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
    vel.peq(acc,dt);
    pos.peq(vel,dt);

    if( printLog )
    {
        cerr<<"EulerSolver, acceleration = "<< acc <<endl;
        cerr<<"EulerSolver, final x = "<< pos <<endl;
        cerr<<"EulerSolver, final v = "<< vel <<endl;
    }
}

void create(EulerSolver*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    obj = new EulerSolver();
    obj->parseFields( arg->getAttributeMap() );
}

SOFA_DECL_CLASS(Euler)

helper::Creator<simulation::tree::xml::ObjectFactory, EulerSolver> EulerSolverClass("EulerSolver");
helper::Creator<simulation::tree::xml::ObjectFactory, EulerSolver> EulerSolverClass2("Euler"); // Previous name for compatibility with existing scenes

} // namespace odesolver

} // namespace component

} // namespace sofa

