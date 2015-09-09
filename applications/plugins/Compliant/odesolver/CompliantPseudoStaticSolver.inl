#include "CompliantPseudoStaticSolver.h"

#include "assembly/AssemblyVisitor.h"

#include <sofa/helper/rmath.h>


namespace sofa {
namespace component {
namespace odesolver {


template< typename CompliantOdeSolver >
CompliantPseudoStaticSolver<CompliantOdeSolver>::CompliantPseudoStaticSolver()
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

template< typename CompliantOdeSolver >
void CompliantPseudoStaticSolver<CompliantOdeSolver>::init()
{
    // clamp velocityFactor [0,1]
    d_velocityFactor.setValue( helper::rclamp<SReal>( d_velocityFactor.getValue(), 0, 1 ) );

    CompliantOdeSolver::init();
}

template< typename CompliantOdeSolver >
void CompliantPseudoStaticSolver<CompliantOdeSolver>::solve(const core::ExecParams* params,
                         SReal dt,
                         core::MultiVecCoordId posId,
                         core::MultiVecDerivId velId)
{
    typename CompliantOdeSolver::SolverOperations sop( params, this->getContext(), this->alpha.getValue(), this->beta.getValue(), dt, posId, velId, true );

    // store previous position (for stop criterion)
    typename CompliantOdeSolver::vec x_prev, x_current;


    const SReal& threshold = d_threshold.getValue();
    const SReal& velocityFactor = d_velocityFactor.getValue();

    unsigned i=0;
    for( unsigned imax=d_iterations.getValue() ; i<imax ; ++i )
    {
        // dynamics integation
        CompliantOdeSolver::solve( params, dt, posId, velId );

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
            x_current.resize( this->sys.m );
            x_prev.resize( this->sys.m );
        }
        this->sys.copyFromMultiVec( x_current, posId ); // get current position
        x_prev -= x_current; // position variation during iteration

        if( x_prev.dot( x_prev ) < threshold*threshold ) break;

        if( this->f_printLog.getValue() )
        {
            serr<<"position variation: "<<sqrt(x_prev.dot( x_prev ))<<sendl;
        }

        x_prev = x_current; // store previous position
    }

    if( this->f_printLog.getValue() ) serr<<i<<" iterations"<<sendl;

}



}
}
}
