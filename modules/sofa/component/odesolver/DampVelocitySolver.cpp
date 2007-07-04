#include <sofa/component/odesolver/DampVelocitySolver.h>
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

int DampVelocitySolverClass = core::RegisterObject("Reduce the velocities")
        .add< DampVelocitySolver >()
        .addAlias("DampVelocity")
        ;

SOFA_DECL_CLASS(DampVelocity);

DampVelocitySolver::DampVelocitySolver()
    : rate( dataField( &rate, 0.99, "rate", "Factor used to reduce the velocities. Typically between 0 and 1.") )
    , threshold( dataField( &threshold, 0.0, "threshold", "Threshold under which the velocities are canceled.") )
{}

void DampVelocitySolver::solve(double dt)
{
    MultiVector vel(this, VecId::velocity());
    bool printLog = f_printLog.getValue();

    if( printLog )
    {
        cerr<<"DampVelocitySolver, dt = "<< dt <<endl;
        cerr<<"DampVelocitySolver, initial v = "<< vel <<endl;
    }

    vel.teq( exp(-rate.getValue()*dt) );
    if( threshold.getValue() != 0.0 )
        vel.threshold( threshold.getValue() );

    if( printLog )
    {
        cerr<<"DampVelocitySolver, final v = "<< vel <<endl;
    }
}

} // namespace odesolver

} // namespace component

} // namespace sofa

