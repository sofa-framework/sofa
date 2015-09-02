#ifndef COMPLIANT_CompliantPseudoStaticSolver_H
#define COMPLIANT_CompliantPseudoStaticSolver_H

#include <Compliant/config.h>
#include <Compliant/odesolver/CompliantImplicitSolver.h>
#include <sofa/simulation/common/MechanicalOperations.h>


namespace sofa {
namespace component {
namespace odesolver {

using core::MultiVecCoordId;
using core::MultiVecDerivId;
using core::ConstMultiVecCoordId;
using core::ConstMultiVecDerivId;

/** (Non-linear) Static solver based on several iterations of dynamics integration.
 *  At each iteration, the velocity can be damped to speed-up convergence (velocityFactor)
 *  The solver stops when the last iteration did not move enough the positions (threshold)
 *  or when the max nb of iterations is reached (iterations).
 *
 * @warning no collision detection is performed at each iteration (only at each step)
 * @warning the system must be constrained enough such as an equilibrium solution exists
 *
 * @author Matthieu Nesme, 2015
 *
*/


class SOFA_Compliant_API CompliantPseudoStaticSolver : public CompliantImplicitSolver {
  public:
				
    SOFA_CLASS(CompliantPseudoStaticSolver, CompliantImplicitSolver);
				
    Data<SReal> d_threshold;        ///< Convergence threshold between 2 iterations
    Data<unsigned> d_iterations;    ///< Max number of iterations
    Data<SReal> d_velocityFactor;        ///< [0,1]  1=fully damped, 0=fully dynamics

    CompliantPseudoStaticSolver();
    virtual ~CompliantPseudoStaticSolver(){}

    virtual void init();

    virtual void solve(const core::ExecParams* params,
                       SReal dt,
                       core::MultiVecCoordId posId,
                       core::MultiVecDerivId velId);

};

}
}
}



#endif
