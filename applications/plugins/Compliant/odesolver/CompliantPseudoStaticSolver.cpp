#include "CompliantPseudoStaticSolver.h"

#include <sofa/core/ObjectFactory.h>

#include "assembly/AssemblyVisitor.h"

#include <sofa/helper/rmath.h>


namespace sofa {
namespace component {
namespace odesolver {


SOFA_DECL_CLASS(CompliantPseudoStaticSolver)
int CompliantPseudoStaticSolverClass = core::RegisterObject("Iterative quasi-static solver")
        .add< CompliantPseudoStaticSolver >();


using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;



CompliantPseudoStaticSolver::CompliantPseudoStaticSolver()
    : d_threshold(initData(&d_threshold,
                   (SReal) 1.0e-6,
                   "threshold",
                   "stop criterion: min difference between 2 iterations"))
    , d_iterations(initData(&d_iterations,
                     (unsigned) 1000,
                     "iterations",
                     "maximum number of iterations"))
    , d_velocityFactor(initData(&d_velocityFactor,
                   (SReal) 0.5,
                   "velocityFactor",
                   "amount of kept velocity at each iteration (0=fully damped, 1=fully dynamics)"))
{}


void CompliantPseudoStaticSolver::init()
{
    // clamp velocityFactor [0,1]
    d_velocityFactor.setValue( helper::rclamp<SReal>( d_velocityFactor.getValue(), 0, 1 ) );

    CompliantImplicitSolver::init();
}


void CompliantPseudoStaticSolver::solve(const core::ExecParams* params,
                         SReal dt,
                         core::MultiVecCoordId posId,
                         core::MultiVecDerivId velId)
{
    SolverOperations sop( params, this->getContext(), alpha.getValue(), beta.getValue(), dt, posId, velId, true );

    // store previous position (for stop criterion)
    vec x_prev, x_current;


    const SReal& threshold = d_threshold.getValue();
    const SReal& velocityFactor = d_velocityFactor.getValue();

    unsigned i=0;
    for( unsigned imax=d_iterations.getValue() ; i<imax ; ++i )
    {
        // dynamics integation
        CompliantImplicitSolver::solve( params, dt, posId, velId );

        // damp velocity
        sop.vop.v_teq( velId, velocityFactor );

        // propagate damped velocity
        {
        simulation::MechanicalPropagatePositionAndVelocityVisitor bob( sofa::core::MechanicalParams::defaultInstance() );
        this->getContext()->executeVisitor( &bob );
        }

        // stop if it does not move enough from previous iteration
        if( !x_current.size() ) // scalar vectors can only be allocated after assembly
        {
            x_current.resize( sys.m );
            x_prev.resize( sys.m );
        }
        sys.copyFromMultiVec( x_current, posId ); // get current position
        x_prev -= x_current; // position variation during iteration

        if( x_prev.dot( x_prev ) < threshold*threshold ) break;

        if( f_printLog.getValue() )
        {
            serr<<"position variation: "<<sqrt(x_prev.dot( x_prev ))<<sendl;
        }

        x_prev = x_current; // store previous position
    }

    if( f_printLog.getValue() ) serr<<i<<" iterations"<<sendl;

}



}
}
}
