#ifndef COMPLIANT_CompliantPseudoStaticSolver_H
#define COMPLIANT_CompliantPseudoStaticSolver_H

#include <Compliant/config.h>
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
 *  The solver stops when the last iteration did not move enough the positions (threshold on velocity norm)
 *  or when the max nb of iterations is reached (iterations).
 *
 * @warning no collision detection is performed at each iteration (only at each step)
 * @warning the system must be constrained enough such as an equilibrium solution exists
 *
 * @author Matthieu Nesme, 2015
 *
*/


template< typename CompliantOdeSolver >
class CompliantPseudoStaticSolver : public CompliantOdeSolver {
  public:
				
    SOFA_CLASS(CompliantPseudoStaticSolver, CompliantOdeSolver);
				
    Data<SReal> d_threshold;        ///< Convergence threshold between 2 iterations (velocity norm)
    Data<unsigned> d_iterations;    ///< Max number of iterations
    Data<SReal> d_velocityFactor;        ///< [0,1]  0=fully damped, 1=fully dynamics

    Data<SReal> d_lastVelocity; ///< output, last velocity square norm

    CompliantPseudoStaticSolver();
    virtual ~CompliantPseudoStaticSolver(){}

    virtual void init();

    virtual void solve(const core::ExecParams* params,
                       SReal dt,
                       core::MultiVecCoordId posId,
                       core::MultiVecDerivId velId);

    virtual std::string getTemplateName() const { return templateName(this); }
    static std::string templateName(const CompliantPseudoStaticSolver<CompliantOdeSolver>* x= NULL) { return CompliantOdeSolver::className( (CompliantOdeSolver*)x ); }
};

}
}
}



#endif
